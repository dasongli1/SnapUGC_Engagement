import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Distortion(nn.Module):
    def __init__(self, n_features=64, future_frames=0, past_frames=0):
        super(Distortion, self).__init__()
        n_feat = n_features
        self.n_feats = n_feat
        kernel_size = 3
        reduction = 4
        bias = False
        act = nn.PReLU()
        self.feat_extract = nn.Sequential(nn.Conv2d(3, self.n_feats, 3, 1, 1),
            CAB(self.n_feats, 3, 4, bias=False, act=nn.PReLU()), CAB(self.n_feats, 3, 4, bias=False, act=nn.PReLU()))
        self.down_1 = nn.Conv2d(self.n_feats, 2*self.n_feats, 3, 2, 1)
        self.encoder1 = [CAB(2*self.n_feats, kernel_size, reduction, bias=bias, act=act) for _ in range(3)]
        self.down_2 = nn.Conv2d(2*self.n_feats, 4*self.n_feats, 3, 2, 1)
        self.encoder2 = [CAB(4*self.n_feats, kernel_size, reduction, bias=bias, act=act) for _ in range(3)]
        self.down_3 = nn.Conv2d(4*self.n_feats, 8*self.n_feats, 3, 2, 1)
        self.encoder3 = [CAB(8*self.n_feats, kernel_size, reduction, bias=bias, act=act) for _ in range(3)]
        self.down_4 = nn.Conv2d(8*self.n_feats, 4*self.n_feats, 3, 2, 1)
        self.encoder4 = [CAB(4*self.n_feats, kernel_size, reduction, bias=bias, act=act) for _ in range(3)]
        self.down_5 = nn.Conv2d(4*self.n_feats, 4*self.n_feats, 3, 2, 1)
        self.encoder5 = [CAB(4*self.n_feats, kernel_size, reduction, bias=bias, act=act) for _ in range(3)]
        #  self.out = nn.Conv2d(16*self.n_feats, 4*self.n_feats, 3, 1, 1)
        self.encoder1 = nn.Sequential(*self.encoder1)
        self.encoder2 = nn.Sequential(*self.encoder2)
        self.encoder3 = nn.Sequential(*self.encoder3)
        self.encoder4 = nn.Sequential(*self.encoder4)
        self.encoder5 = nn.Sequential(*self.encoder5)
        # self.avgpool = 
        # self.fc = 
        num_classes = 25
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(64*4, 64*2), nn.PReLU(), nn.Linear(64*2,64*2), nn.PReLU(), nn.Linear(64*2,64*2), nn.PReLU(), nn.Linear(64*2, num_classes))
        self.fc2 = nn.Sequential(nn.Linear(64*4, 64*2), nn.PReLU(), nn.Linear(64*2,64*2), nn.PReLU(), nn.Linear(64*2,64*2), nn.PReLU(), nn.Linear(64*2,1))
        self.fc3 = nn.Sequential(nn.Linear(64*4, 64*2), nn.PReLU(), nn.Linear(64*2,64*2), nn.PReLU(), nn.Linear(64*2,64*2), nn.PReLU(), nn.Linear(64*2,1))
        
    def forward(self, x):
        x = self.feat_extract(x)
        x = self.down_1(x)
        x = self.encoder1(x)
        x = self.down_2(x)
        x = self.encoder2(x)
        x = self.down_3(x)
        x = self.encoder3(x)
        x = self.down_4(x)
        x = self.encoder4(x)
        x = self.down_5(x)
        x = self.encoder5(x)
        # print(x.shape)
        temp = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return temp, x1, x2, x3
        
        
