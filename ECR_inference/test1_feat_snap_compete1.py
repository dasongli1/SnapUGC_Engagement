import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import tensorflow_hub as hub
import numpy as np
from modules.distort import Distortion 
from modules.efficientnet_v2 import EfficientNetV2
from data.video_patch_image_dataset import VideoImageDataset
from modules.resnet3d import generate_model
import cv2
import csv
from pathlib import Path
from torch.nn import functional as F
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
import scipy
from video_downstream_datasets import read_data, read_data2
from ruamel.yaml import YAML
import sys
sys.path.append("./mPLUG_2/")
from models.model_video_caption_mplug2 import MPLUG2
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer
from modules.EVQA import EVQA
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", cache_dir = "/scratch/gilbreth/li5823/cache_huggingface/"
    )
model = EVQA(3, 16, pipe.tokenizer, pipe.text_encoder)

model.load_state_dict(torch.load("checkpoints/EVQA.pth")['params'])
print("loading successful")
model.eval()
model.cuda()

config = "mPLUG_2/configs_video/VideoCaption_msrvtt_large2.yaml"
path = Path(config)
yaml = YAML(typ='safe')
config = yaml.load(path)
config["min_length"] = 4
config["max_length"] = 20
config["add_object"] = False
config["beam_size"] = 5
config['text_encoder'] = "bert-large-uncased"
config['text_decoder'] = "bert-large-uncased"

config['temporal_stride'] = 2
config['temporal_downsampling'] = False
config['accum_steps'] = 2
config['no_randaug'] = False

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    #### Model ####
print("Creating model")
captioning_model = MPLUG2(config=config, tokenizer=tokenizer)
device = torch.device("cuda")
captioning_model = captioning_model.to(device)
checkpoint = torch.load("checkpoints/mPLUG2_MSRVTT_Caption.pth", map_location='cpu')
try:
    state_dict = checkpoint['model']
except:
    state_dict = checkpoint['module']
num_patches = int(224 *224/(14*14))
pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
captioning_model.load_state_dict(state_dict, strict=False)

model_music = hub.load('https://tfhub.dev/google/yamnet/1')
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = model_music.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

model2 = Distortion()
model2.load_state_dict(torch.load("checkpoints/net_distort6_g_latest.pth")['params'])
model2.cuda()
model2.eval()

model1 = EfficientNetV2('s',
                        in_channels=3,
                        n_classes=50,
                        pretrained=True)
model1.cuda()
model1.eval()

model3 = generate_model(18)
model3 = model3.cuda()
model3.load_state_dict(torch.load("checkpoints/r3d18_K_200ep.pth")['state_dict'])
model3.eval()



videos_dir = "/scratch/gilbreth/li5823/test_videos/test_videos/"
videos_files = "dataset/val_data.csv"
dataset = VideoImageDataset(videos_files, videos_dir)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,collate_fn=None,pin_memory=False)
n_seq = 16
normalize = torchvision.transforms.Normalize(0.5, 0.5)
avg = nn.AdaptiveAvgPool2d((1,1))
avg1= nn.AdaptiveAvgPool3d((1,1,1))
avg2= nn.AdaptiveAvgPool3d((None,1,1))
num_one_running = 48
mos_label_list = []
dmos_out_list = []
def feat2_func(model):
    feat_1_out = []
    feat_x_out = []
    video_new_len = tensors.shape[0]
        
    video_residual = video_new_len % num_one_running
    for kkk in range(video_new_len // num_one_running):
        feat_x, _, feat_1, _ = model(tensors[kkk*num_one_running: (kkk+1)*num_one_running])
        feat_1_out.append(feat_1[:,0])
        feat_x_out.append(avg(feat_x))
    if video_residual != 0:
        feat_x, _, feat_1, _ = model(tensors[-video_residual:])
        feat_1_out.append(feat_1[:,0])
        feat_x_out.append(avg(feat_x))
    feat_1_out = torch.cat(feat_1_out, dim=0)
    feat_x_out = torch.cat(feat_x_out, dim=0)
    return feat_x_out
def feat1_func(model):
    # tensors = normalize(tensors)
    feat_1_out = []
    # continue
    video_new_len = tensors.shape[0]
    video_residual = video_new_len % num_one_running
    for kkk in range(video_new_len // num_one_running):
        feat_1_list = []
        features = model.get_features(normalize(tensors[kkk*num_one_running: (kkk+1)*num_one_running]))
        # print(feat_1.shape)
        for i, feature in enumerate(features):
            # print('feature %d shape:' % i, feature.shape)
            feat_1_list.append(avg(feature))
        feat_1 = torch.cat((feat_1_list), dim=1)
        # print(feat_1.shape)
        feat_1_out.append(feat_1[:,:])
    
    if video_residual != 0:
        feat_1_list = []
        features = model.get_features(tensors[-video_residual:])
        for i, feature in enumerate(features):
            feat_1_list.append(avg(feature))
        feat_1 = torch.cat((feat_1_list), dim=1)
        feat_1_out.append(feat_1[:,:])
    feat_1_out = torch.cat(feat_1_out, dim=0)
    return feat_1_out
def feat3_func(model):
    feat_out = []
    video_new_len = tensors.shape[0]
    for kkk in range(video_new_len // n_seq):
        feat_1_list = []
        temp_input = tensors[kkk*n_seq:(kkk+1)*n_seq]
        # print(temp_input.shape)
        temp_input = temp_input.unsqueeze(0).permute(0,2,1,3,4)
        video_out = avg1(model(temp_input)).squeeze(2)
        # print(temp_input.shape, video_out.shape)
        # exit(0)
        feat_out.append(video_out)
    feat_out = torch.cat(feat_out, dim=0)
    # print(feat_out.shape, tensors.shape)
    # exit(0)
    return feat_out
def feat4_func(model):
    feat_out = []
    video_new_len = tensors_ori.shape[0]
    print(tensors_ori.shape)
    # exit(0)
    feat_out = avg2(model(tensors_ori.unsqueeze(0).permute(0,2,1,3,4)))
    # print(feat_out.shape, tensors.shape, tensors.shape, tensors_ori.unsqueeze(0).permute(0,2,1,3,4).shape)
    # exit(0)
    for kkk in range(video_new_len // n_seq):
        feat_1_list = []
        temp_input = tensors[kkk*n_seq:(kkk+1)*n_seq]
        # print(temp_input.shape)
        temp_input = temp_input.unsqueeze(0).permute(0,2,1,3,4)
        video_out = avg1(model(temp_input)).squeeze(2)
        feat_out.append(video_out)
    feat_out = torch.cat(feat_out, dim=0)
    return feat_out[0].permute(1,0,2,3)
def captioning(path):
    if True:
        path = video_name[0]
        array_ = read_data(path, num_frame=32)
        array_2 = read_data2(path).unsqueeze(0)
        if True:
            videos = array_2
            b, c, n, h, w = videos.shape
            n_seq = 16
            n_seq_multiple = n // n_seq
            n_seq_residual = n % n_seq
            videos = videos.cuda()

            if n_seq_residual > 3:
                videos = torch.cat((videos[:,:,0:n_seq_multiple*n_seq], videos[:,:,-n_seq:]), dim=2)
            elif n_seq_residual != 0:
                videos = videos[:,:,0:n_seq_multiple*n_seq]
            # print(videos.shape)
            # exit(0)
            video_new_len = videos.shape[2]
            videos = videos.permute(0,2,1,3,4)[0].view(video_new_len//n_seq, n_seq, c, h, w)
            # print(videos.shape)
            # exit(0)
            image_embeds_list = []
            bs = 4
            for kkk in range(video_new_len // n_seq // bs):
                video = videos[kkk*bs:(kkk+1)*bs,:].permute(0,2,1,3,4)
                # video = video.to(device,non_blocking=True)
                # topk_ids, topk_probs, image_embeds = model(video, train=False)
                image_embeds = captioning_model.forward_feature(video, train=False)
                image_embeds_list.append(image_embeds)
            if (kkk+1)* bs < video_new_len // n_seq:
                video = videos[(kkk+1)*bs:,:].permute(0,2,1,3,4)
                image_embeds = captioning_model.forward_feature(video, train=False)
                image_embeds_list.append(image_embeds)
            image_embeds_all = torch.cat(image_embeds_list, dim=0)
            image_embeds_all = torch.mean(image_embeds_all, dim=1)
        array_ = array_.to(device,non_blocking=True).unsqueeze(0)
        topk_ids, topk_probs = captioning_model(array_, train=False)
        for topk_id, topk_prob in zip(topk_ids, topk_probs):
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            ans = ans.split(config["prompt"])[-1].strip()
    return image_embeds_all, ans
def sound_classification(out_path):
    if True:        
        out_path = path[0:-4]+".wav"
        cmd = "/depot/chan129/users/dasongli/ffmpeg-git-20240629-amd64-static/ffmpeg -v quiet -y -i %s -ac 1 -f wav %s"%(path, out_path)
        # print(cmd)
        os.system(cmd)
        speech_name = out_path
        sample_rate, wav_data = wavfile.read(speech_name, 'rb')
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
        duration = len(wav_data)/sample_rate
        # print(speech_name, f'Total duration: {duration:.2f}s')
        Audio(wav_data, rate=sample_rate)
        waveform = wav_data / tf.int16.max
        scores, embeddings, spectrogram = model_music(waveform)
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()
        scores_np = scores_np.mean(axis=0)
        scores_topk = scores_np.argsort()[-5:][::-1]
        infered_class1 = class_names[scores_topk[0]]
        infered_class2 = class_names[scores_topk[1]]
        infered_class3 = class_names[scores_topk[2]]
        infered_class4 = class_names[scores_topk[3]]
        infered_class5 = class_names[scores_topk[4]]
        music1_text = f'{infered_class1}, {infered_class2}, {infered_class3}, {infered_class4}, {infered_class5}'
        os.remove(out_path)
    return music1_text
out_list = []
id_list = []
with torch.no_grad():
    for idx, samples in enumerate(data_loader):
        tensors, video_name, text1, text2= samples
        if isinstance(tensors, str):
            print("failed", video_name)
            continue

        path = video_name[0]
        name = path.split("/")[-1].split(".")[0]
        image_embeds_all, caption = captioning(path)
        music1_text = sound_classification(path)

        video_name = video_name[0]
        video_id = video_name.split("/")[-1].split(".")[0]
        tensors = tensors[0].permute(0,3,1,2)
        _, _, H, W = tensors.shape
        if H < W:
            tensors = tensors.permute(0,1,3,2)
        tensors = F.interpolate(tensors, (452, 256), mode="bicubic")
        tensors = tensors.cuda()
        tensors = torch.flip(tensors, (1,))
        tosave_img = tensors[0].permute(1,2,0).cpu().data.numpy()
        video_len, c, h, w = tensors.shape
        n_seq_multiple = video_len // n_seq
        n_seq_residual = video_len % n_seq
            
        tensors_ori = tensors.clone()
        if n_seq_residual > 3:
            tensors = torch.cat((tensors[0:n_seq_multiple*n_seq], tensors[-n_seq:]), dim=0)
        elif n_seq_residual != 0:
            tensors = tensors[0:n_seq_multiple*n_seq]
        tosave_name = video_name.split("/")[-1][0:-4] 
        feat2_out = feat2_func(model2)
        feat1_out = feat1_func(model1)
        feat3_out = feat3_func(model3)
        feat_4 = image_embeds_all.unsqueeze(0)
        out0 = model([music1_text], [text1[0]], [text2[0]], [caption], feat1_out, feat2_out, feat3_out, image_embeds_all)
        
        out0_mean = torch.mean(out0)
        out0_val = out0_mean.clamp(0.0, 1.0).item()
        # mos0_val = mos_label[0].clamp(0.0,1.0).item() # / 20.0
        
        # mos0_labels.append(mos0_val)
        id_list.append(video_id)
        out_list.append(float(out0_val))
        # c0_list.append(video_len)
        if idx % 20 == 0:
            print(idx, video_id, out0_val)
        with open("result_test2_captioning3R_allfeat_val_80w.txt", "a") as f:
            f.write(str(video_id)+"\t"+str(out0_val)+"\n") 
with open("submission_baseline.csv", "w", newline="") as csvfile:
    fieldnames = ['Id', 'ECR']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(id_list)):
        dict_ = {'Id': id_list[i], 'ECR': out_list[i]}
        writer.writerow(dict_)
