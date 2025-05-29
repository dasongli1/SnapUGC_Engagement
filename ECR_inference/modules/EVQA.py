import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import torchvision
import random
from typing import Optional, Union
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape, q.shape, k.shape)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        # print(attn.shape, q.shape, k.shape, x.shape)
        # exit(0)
        return x
        # exit(0)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        #q = torch.nn.functional.normalize(q, dim=-1)
        #k = torch.nn.functional.normalize(k, dim=-1)

        #attn = (q @ k.transpose(-2, -1)) * self.temperature
        #attn = attn.softmax(dim=-1)

        #out = (attn @ v)
        
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Mlp(dim,  hidden_features=dim*ffn_expansion_factor, out_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        processor: Optional["AttnProcessor"] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        processor = processor if processor is not None else CrossAttnProcessor()
        self.set_processor(processor)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = CrossAttnAddedKVProcessor()
        else:
            processor = CrossAttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor"):
        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `CrossAttention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length):
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        return attention_mask


class CrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        # print("init", hidden_states.shape, encoder_hidden_states.shape)
        # exit(0)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)
        # print("query:", query.shape)
        query = attn.head_to_batch_dim(query)
        # print(query.shape) # [128, 1, 64] 
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # print(key.shape, value.shape)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # print(key.shape, value.shape) # [8,77,64] 

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # print(query.shape, key.shape, value.shape, attention_probs.shape, hidden_states.shape)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class Linear_block(nn.Module):
    def __init__(self, in_size, relu_slope=0.2):
        super(Linear_block, self).__init__()
        out_size = in_size
        self.conv_1 = nn.Linear(in_size, out_size) # nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Linear(out_size, out_size) # nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        # self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.conv_2(out)
        return x + out

class EVQA(nn.Module):
    def __init__(self, in_c=3, n_seq=16, tokenizer=None, text_encoder=None):
        super(EVQA, self).__init__()
        self.conv = conv(in_c, in_c, 3)
        self.n_seq = n_seq
        num_class = 256
        self.fc1 = nn.Sequential(nn.Linear(528, num_class), nn.LeakyReLU(0.2), Linear_block(num_class), Linear_block(num_class), Linear_block(num_class))
        #self.fc2 = nn.Sequential(nn.Linear(528, num_class), nn.LeakyReLU(0.2), nn.Linear(num_class, num_class), nn.LeakyReLU(0.2), nn.Linear(num_class, num_class), nn.LeakyReLU(0.2), nn.Linear(num_class, num_class), nn.LeakyReLU(0.2), nn.Linear(num_class, num_class))
        self.fc3 = nn.Sequential(nn.Linear(256, num_class), nn.LeakyReLU(0.2), Linear_block(num_class), Linear_block(num_class), Linear_block(num_class))
        # self.fc4 = nn.Sequential(nn.Linear(1024, num_class), nn.LeakyReLU(0.2), nn.Linear(num_class, num_class), nn.LeakyReLU(0.2), nn.Linear(num_class, num_class), nn.LeakyReLU(0.2), nn.Linear(num_class, num_class), nn.LeakyReLU(0.2), nn.Linear(num_class, num_class))
        self.fc4 = nn.Sequential(nn.Linear(1024, num_class*4), nn.LeakyReLU(0.2), Linear_block(num_class*4), Linear_block(num_class*4),  Linear_block(num_class*4), Linear_block(num_class*4))
        self.fc30 = nn.Sequential(nn.Linear(512, num_class*2), nn.LeakyReLU(0.2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2))
        self.feat3_preprocess = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(0.2), Linear_block(512), Linear_block(512), Linear_block(512), Linear_block(512))
        self.fc_merge12 = nn.Linear(num_class*(self.n_seq)*2, num_class*4)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.num_class = num_class
        self.fc_merge123 = nn.Sequential(nn.Linear(num_class*4+512+512+512+512+512+1024, num_class*2), nn.LeakyReLU(0.2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2)) # , nn.LeakyReLU(0.2), nn.Linear(num_class*2, 1))
        self.block1 = nn.Sequential(TransformerBlock(2*num_class, 8, 2),TransformerBlock(2*num_class, 8, 2),TransformerBlock(2*num_class, 8, 2), TransformerBlock(2*num_class, 8, 2), TransformerBlock(2*num_class, 8, 2), TransformerBlock(2*num_class, 8, 2),TransformerBlock(2*num_class, 8, 2), TransformerBlock(2*num_class, 8, 2))
        # self.out = nn.Sequential(nn.Linear(num_class*2, num_class*2), nn.LeakyReLU(0.2), Linear_block(num_class*2))
        # self.out2 = nn.Sequential(nn.Linear(num_class*2, num_class*2), nn.LeakyReLU(0.2), nn.Linear(num_class*2, 1))
        self.out = nn.Sequential(nn.Linear(num_class*2, num_class*2), nn.LeakyReLU(0.2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2), nn.Linear(num_class*2, 1))
        self.out0 = nn.Sequential(nn.Linear(num_class*2, num_class*2), nn.LeakyReLU(0.2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2), nn.Linear(num_class*2, 1))
        self.out_1 = nn.Sequential(nn.Linear(num_class*2+3, num_class*2), nn.LeakyReLU(0.2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2), nn.Linear(num_class*2, 1))
        self.out0_1 = nn.Sequential(nn.Linear(num_class*2+3, num_class*2), nn.LeakyReLU(0.2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2), Linear_block(num_class*2), nn.Linear(num_class*2, 1))
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.cross_attn1 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
        self.cross_attn1_0 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
        self.self_attn1_0 = Linear_block(num_class * 2)
        self.cross_attn4 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
        self.cross_attn4_0 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
        self.self_attn4_0 = Linear_block(num_class * 2)
        self.cross_attn2 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
        self.cross_attn2_0 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
        self.self_attn2_0 = Linear_block(num_class * 2)
        self.cross_attn2_1 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
        self.cross_attn3 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
        self.cross_attn3_0 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
        self.self_attn3_0 = Linear_block(num_class * 2)
        self.cross_attn3_1 = CrossAttention(query_dim = num_class * 2, cross_attention_dim=768)
    def forward(self, music_text1, music_text2, music_text3, music_text4, feat1, feat2, feat3, feat4):
        with torch.no_grad():
            shape3 = feat3.shape[0]
            shape4 = feat4.shape[0]
            if shape3 > shape4:
                feat3 = feat3[0:shape4]
                feat1 = feat1[0:shape4*16]
                feat2 = feat2[0:shape4*16]
            elif shape3 < shape4:
                feat4 = feat4[0:shape3]
            text_input1 = self.tokenizer(
                music_text1,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embedding1 = self.text_encoder(text_input1.input_ids.cuda())[0]
            text_embedding1 = text_embedding1.repeat(feat3.shape[0],1,1)
            
            text_input2 = self.tokenizer(
                music_text2,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embedding2 = self.text_encoder(text_input2.input_ids.cuda())[0]
            text_embedding2 = text_embedding2.repeat(feat3.shape[0],1,1)
            
            text_input3 = self.tokenizer(
                music_text3,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embedding3 = self.text_encoder(text_input3.input_ids.cuda())[0]
            text_embedding3 = text_embedding3.repeat(feat3.shape[0],1,1)
            
            text_input4 = self.tokenizer(
                music_text4,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embedding4 = self.text_encoder(text_input4.input_ids.cuda())[0]
            text_embedding4 = text_embedding4.repeat(feat3.shape[0],1,1)
            
        feat_1 = feat1.squeeze(3).squeeze(2)
        feat_2 = feat2.squeeze(3).squeeze(2)
        feat_3 = feat3.squeeze(3).squeeze(2)
        feat_4 = feat4
        feat_3_embed = self.feat3_preprocess(feat_3)
        feat_3 = self.fc30(feat_3)
        feat_4 = self.fc4(feat_4)
        b, c = feat_1.shape
        temp = torch.cat((self.fc1(feat_1), self.fc3(feat_2)), dim=1)
        temp = temp.view(b // (self.n_seq), self.n_seq * self.num_class * 2)
        feat_12 = self.fc_merge12(temp)
        music_feature1 = self.cross_attn1(feat_3_embed.unsqueeze(1), text_embedding1).squeeze(1)
        music_feature1 = self.cross_attn1_0(self.self_attn1_0(music_feature1).unsqueeze(1), text_embedding1).squeeze(1)
        music_feature4 = self.cross_attn4(feat_3_embed.unsqueeze(1), text_embedding4).squeeze(1)
        music_feature4 = self.cross_attn4_0(self.self_attn4_0(music_feature4).unsqueeze(1), text_embedding4).squeeze(1)
        music_feature2 = self.cross_attn2(feat_3_embed.unsqueeze(1), text_embedding2).squeeze(1)
        music_feature2 = self.cross_attn2_0(self.self_attn2_0(music_feature2).unsqueeze(1), text_embedding2)# .squeeze(1)
        music_feature3 = self.cross_attn3(feat_3_embed.unsqueeze(1), text_embedding3).squeeze(1)
        music_feature3 = self.cross_attn3_0(self.self_attn3_0(music_feature3).unsqueeze(1), text_embedding3)# .squeeze(1)
        music_feature2_ = self.cross_attn2_1(music_feature3, text_embedding2).squeeze(1)
        music_feature3_ = self.cross_attn3_1(music_feature2, text_embedding3).squeeze(1)
        
        feat123 = self.fc_merge123(torch.cat((feat_12, feat_3, feat_4, music_feature1, music_feature2_, music_feature3_, music_feature4), dim=1))
        feat123 = feat123.unsqueeze(0)
        out = self.block1(feat123).squeeze(0)
        temp_out = self.out(out)
        temp_out = torch.mean(temp_out, dim=0)
        return temp_out
