import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.utils.data as data

from torch.utils.data import Dataset
from PIL import Image
import re

class SegmentationMaskDataset(Dataset):
    def __init__(self, root_dir, split = 'train', n_frames = 10):
        """returns dataset with n_frames sliding window len data

        Args:
            root_dir (_type_): dir containing train and valid folders
            split (str, optional): train or valid. Defaults to 'train'.
            n_frames (int, optional): _description_. Defaults to 10. num_frames / 2 for future and past each
        """
        
        self.n_frames = n_frames
        self.map_idx_image_folder = []
        self.data_dir = os.path.join(root_dir, split)
        self.split = split
        self.per_vid_data_len = 22 - n_frames + 1
        for v in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, v)):
                self.map_idx_image_folder.append(os.path.join(self.data_dir, v))
        
    def __len__(self):
          return len(self.map_idx_image_folder * self.per_vid_data_len )
    
    def resize_to_224(self, mask):
        # pad 40 x 240 up and down
        zeros = torch.zeros((self.n_frames,40,240), dtype = torch.long)
        mask = torch.cat((zeros,mask,zeros), dim= 1)
        return mask
    
    def __getitem__(self, idx):
        # if self.split == "train": # return initital 11 frame only
        video_num = idx // self.per_vid_data_len
        start_idx = idx % self.per_vid_data_len
        
        all_frame_mask = torch.FloatTensor(np.load(os.path.join(self.map_idx_image_folder[video_num], 'mask.npy')))[start_idx: start_idx + self.n_frames]
        
        all_frame_mask[all_frame_mask >= 49] = 0 # set some noise to background
        # do this padding to make image_size = 240
        all_frame_mask = self.resize_to_224(all_frame_mask)

        return all_frame_mask, torch.tensor(1) # return dummy one as target


class ElevenVsOneFramePredDatasets(Dataset):
    def __init__(self, root_dir, split = 'train', mode = 'cont', tranforms = None):
        self.map_idx_image_folder = []
        self.mode = mode
        self.data_dir = os.path.join(root_dir, split)
        self.split = split
        self.per_vid_data_len = 11
        self.transforms = tranforms
        for v in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, v)):
                self.map_idx_image_folder.append(os.path.join(self.data_dir, v))
    
    def __len__(self):
        if self.mode == 'last':
            return len(self.map_idx_image_folder  )
        return len(self.map_idx_image_folder  ) * self.per_vid_data_len
    
    def __getitem__(self, idx):
        # if self.split == "train": # return initital 11 frame only
        video_num = idx // self.per_vid_data_len
        start_idx = idx % self.per_vid_data_len
        if self.mode == 'last':
            video_num = idx
            start_idx = 0
        
        req_image_idx= [start_idx + i for i in range(0,11)]

        if self.mode == 'last':
            #req_image_idx.append(21)
            req_image_idx= [start_idx + i for i in range(0,22)]
        else:
            req_image_idx.append(start_idx + 11) # add 12 th frame

        images = []
        pattern = re.compile(r'video_(\d+)$')
        #video_number = int(match.group(1))
        match = pattern.search(self.map_idx_image_folder[video_num])
        video_number = int(match.group(1))
        for i in req_image_idx:
            img_path = os.path.join(self.map_idx_image_folder[video_num], f"image_{i}.png" )
            image = Image.open(img_path)

            if self.transforms:
                image = self.transforms(image)
            images.append(image)

        return torch.stack(images), torch.tensor(video_number)

    


class NextFramePredDatasets(Dataset):
    def __init__(self, root_dir, split = 'train', mode = '5v6', tranforms = None):
        self.map_idx_image_folder = []
        self.mode = mode
        self.data_dir = os.path.join(root_dir, split)
        self.split = split
        self.transforms = tranforms
        for v in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, v)):
                self.map_idx_image_folder.append(os.path.join(self.data_dir, v))
    
    def __len__(self):
          return len(self.map_idx_image_folder  )
    
    def __getitem__(self, idx):


        if self.mode =='5v6':
            req_image_idx = [1 + x for x in range(0, 21, 2)]
        
        elif self.mode == '11v1':
            req_image_idx = list(range(0,12))
            req_image_idx.append(21)
        
        else:
            raise Exception("mode not right for data pulling")

        images = []

        pattern = re.compile(r'video_(\d+)$')
        #video_number = int(match.group(1))
        match = pattern.search(self.map_idx_image_folder[idx])
        video_number = int(match.group(1))

        for i in req_image_idx:
            img_path = os.path.join(self.map_idx_image_folder[idx], f"image_{i}.png" )
            image = Image.open(img_path)

            if self.transforms:
                image = self.transforms(image)
            images.append(image)

        return torch.stack(images), torch.tensor(video_number)
    











