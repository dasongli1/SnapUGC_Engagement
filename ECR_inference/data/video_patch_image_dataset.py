# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

#from basicsr.data.data_util import np2Tensor, get_patch, data_augment
#from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
#from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import os
import glob, imageio
import numpy as np
import torch
import csv
import cv2

def scan(d, feat4_dir):
    name_list = []
    label4_list = []
    label5_list = []
    frame_len_list = []
    with open(d, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = os.path.join(feat4_dir, row['Id']+".mp4")
            name_list.append(name)
            label4_list.append(row['Title'])
            label5_list.append(row['Description'])

    return name_list, label4_list, label5_list

class VideoImageDataset(data.Dataset):
    def __init__(self, video_dir, filename):

        self.n_frames_video = []
        self.name_list, self.label4_list, self.label5_list = scan(video_dir, filename)
    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'gt')
        self.dir_input = os.path.join(self.apath, 'blur')
        print("DataSet GT path:", self.dir_gt)
        print("DataSet INPUT path:", self.dir_input)

    def _scan(self):
        vid_gt_names = sorted(glob.glob(os.path.join(self.dir_gt, '*')))
        vid_input_names = sorted(glob.glob(os.path.join(self.dir_input, '*')))
        assert len(vid_gt_names) == len(vid_input_names), "len(vid_gt_names) must equal len(vid_input_names)"

        images_gt = []
        images_input = []

        for vid_gt_name, vid_input_name in zip(vid_gt_names, vid_input_names):
            # if self.train:
            gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))[:self.n_frames_per_video]
            input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))[:self.n_frames_per_video]
            # else:
            #    gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))
            #     input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
            images_gt.append(gt_dir_names)
            images_input.append(input_dir_names)
            self.n_frames_video.append(len(gt_dir_names))

        return images_gt, images_input

    def _load(self, images_gt, images_input):
        data_input = []
        data_gt = []

        n_videos = len(images_gt)
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" % idx)
            gts = np.array([imageio.imread(hr_name) for hr_name in images_gt[idx]])
            inputs = np.array([imageio.imread(lr_name) for lr_name in images_input[idx]])
            data_input.append(inputs)
            data_gt.append(gts)

        return data_gt, data_input

    def __getitem__(self, idx):
        video_name = self.name_list[idx]
        frames = self._load_file(idx)
        title = self.label4_list[idx]
        description = self.label5_list[idx]
        return torch.FloatTensor(frames) / 255.0, video_name, title, description

    def __len__(self):
        return len(self.name_list)

    def _get_index(self, idx):
        #if self.train:
        return idx % self.num_frame
        #else:
        #    return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        frames = []
        video_name = self.name_list[idx] 
        
        vidcap = cv2.VideoCapture(video_name)
        success, frame = vidcap.read()
        while success:
            frames.append(frame)
            success, frame = vidcap.read()
        frame_len = len(frames)
        frames = np.stack(frames)
        return frames

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        gts = self.data_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        inputs = self.data_input[video_idx][frame_idx:frame_idx + self.n_seq]
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]]

        return inputs, gts, filenames

    def get_patch(self, input, gt, size_must_mode=1):
        if True:
            input, gt = get_patch(input, gt, patch_size=self.patch_size)
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
            if not self.no_augment:
                input, gt = data_augment(input, gt)
        return input, gt
