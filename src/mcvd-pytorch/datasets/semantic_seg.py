import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.utils.data as data



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
        for v in os.listdir():
            if os.path.isdir(os.path.join(self.data_dir, v)):
                self.map_idx_image_folder.append(os.path.join(self.data_dir, v))
        
    def __len__(self):
          return len(self.map_idx_image_folder * self.per_vid_data_len )
    

    def __getitem__(self, idx):
        # if self.split == "train": # return initital 11 frame only
        video_num = (idx + 1) // self.per_vid_data_len
        start_idx = idx % self.per_vid_data_len
        
        all_frame_mask = torch.FloatTensor(np.load(os.path.join(self.map_idx_image_folder[video_num], 'mask.npy'))).long()[start_idx: start_idx + self.n_frames]
        all_frame_mask[all_frame_mask >= 49] = 0 # set some noise to background
        
        return all_frame_mask, torch.tensor(1) # return dummy one as target