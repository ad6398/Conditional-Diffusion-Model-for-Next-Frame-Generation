from load_model_from_ckpt import load_model, get_sampler, init_samples
from datasets import get_dataset, data_transform, inverse_data_transform
from runners.ncsn_runner import conditioning_fn
from models import ddpm_sampler

import glob, os
#import mediapy as media
import torch
from torch.utils.data import DataLoader, Dataset

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import re
from PIL import Image
from functools import partial
from tqdm import tqdm
# import wandb

class TestElevenVsOneFramePredDatasets(Dataset):
    def __init__(self, root_dir, start=1000, end= 2000, split = 'val',  tranforms = None):
        self.map_idx_image_folder = []
        #self.mode = mode
        ignore_list = [1117,1119,1121,1126,1313,1314,1325,1542,1545,1573, 1587,1589,1741,1748, 1770,1777,1783,1784,1922,1925, 1100, 1580, 1779, 1913, 1349, 1574, 1128, 1322, 1746, 1914]
        self.data_dir = os.path.join(root_dir, split)
        self.split = split
        self.per_vid_data_len = 11
        self.transforms = tranforms
        for v in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, v)):
                pattern = re.compile(r'video_(\d+)$')
                match = pattern.search(v)
                video_number = int(match.group(1))
                if video_number < end and video_number >=start:
                  self.map_idx_image_folder.append(os.path.join(self.data_dir, v))

        

    def __len__(self):
        return len(self.map_idx_image_folder )

    def __getitem__(self, idx):
        # if self.split == "train": # return initital 11 frame only

        video_num = idx
        start_idx = 0

        req_image_idx= [start_idx + i for i in range(0,11)]

        if self.split == 'val':
            req_image_idx.append(21)
        else:
            req_image_idx.append(0)

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

def stretch_image(X, ch, imsize):
    return X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 2, 1, 4, 3).reshape(len(X), ch, -1, imsize).permute(0, 1, 3, 2)

def predict_one_frame_autoregressive(data_path, ckpt_path, start, end, output_dir):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    scorenet, config = load_model(ckpt_path, device)
        # wandb_run = wandb.init(project= "DL-next-fram-11v1-test", config = config )
    sampler = partial(ddpm_sampler, config=config)
    test_transform = transforms.Compose([
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor()
            ])
    test_dataset = TestElevenVsOneFramePredDatasets(root_dir=data_path, start= start, end = end, split= 'hidden', tranforms= test_transform )
    num_frames_pred = config.sampling.num_frames_pred
    test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False,
                         num_workers=config.data.num_workers, drop_last=False)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # scorenet, config = load_model(ckpt_path, device)
    # # wandb_run = wandb.init(project= "DL-next-fram-11v1-test", config = config )
    # sampler = partial(ddpm_sampler, config=config)
    # test_transform = transforms.Compose([
    #         transforms.Resize((config.data.image_size, config.data.image_size)),
    #         transforms.ToTensor()
    #     ])

    # test_dataset = TestElevenVsOneFramePredDatasets(root_dir=data_path,split= 'val', tranforms= test_transform )
    # num_frames_pred = config.sampling.num_frames_pred

    # test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False,
    #                      num_workers=config.data.num_workers, drop_last=False)
    # print(" dev and config dev ", device, config.device)
    avg_mse = 0.0
    image_transform = transforms.Compose([
        transforms.ToPILImage()])
    for i, (test_x, video_num) in enumerate(test_loader):
        # print("pred", i)
        test_x = data_transform(config, test_x)
        # eliminate conditioning_fn
        real, cond, cond_mask = conditioning_fn(config, test_x, num_frames_pred=config.data.num_frames,
                                        prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
                                        prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0))

        real = inverse_data_transform(config, real)
        cond = cond.to(config.device)
        init_samples_shape = (real.shape[0], config.data.channels*config.data.num_frames,
                                  config.data.image_size, config.data.image_size)
        init_samples = torch.randn(init_samples_shape, device=config.device)
        n_iter_frames = 11

        pred_samples = []
        last_frame = None
        print("real, cond, init_sample shape", real.shape, cond.shape, init_samples_shape)
        for i_frame in tqdm(range(n_iter_frames), desc="Generating video frames"):
            gen_samples = sampler(init_samples,
                                        scorenet, cond=cond, cond_mask=cond_mask,
                                        n_steps_each=config.sampling.n_steps_each,
                                        step_lr=config.sampling.step_lr,
                                        verbose=True,
                                        final_only=True,
                                        denoise=config.sampling.denoise,
                                        subsample_steps=1000,
                                        clip_before=getattr(config.sampling, 'clip_before', True),
                                        t_min=getattr(config.sampling, 'init_prev_t', -1),
                                        gamma=getattr(config.model, 'gamma', False))

            gen_samples = gen_samples[-1].reshape(gen_samples[-1].shape[0], config.data.channels*config.data.num_frames,
                                                        config.data.image_size, config.data.image_size)
            # print("gen sample shape", gen_samples.shape)
            pred_samples.append(gen_samples.to('cpu'))
            # print("len pred sample",len(pred_samples))
            last_frame = gen_samples.to('cpu')
            if i_frame == n_iter_frames - 1:
                last_frame = gen_samples.to('cpu')
                continue

            # init
            cond = torch.cat([cond[:, config.data.channels:], gen_samples[:, :config.data.channels]], dim=1)
            init_samples = torch.randn(init_samples_shape, device=config.device)

        # only last frame
        pred = torch.cat(pred_samples, dim=1)
        # print("pred shape before", pred.shape)
        # pred = torch.cat(pred_samples, dim=1)[:, :config.data.channels*num_frames_pred]
        pred = inverse_data_transform(config, pred)
        # print("pred shape", pred.shape)
        last_frame = inverse_data_transform(config, last_frame)
        # print("last fram shape", last_frame.shape)
        out_folder_name = os.path.join(output_dir, f"11v1-cont-{start}-{end}-val-pred")
        if not os.path.exists(out_folder_name):
          os.mkdir(out_folder_name)
        for idx in range(video_num.shape[0]):
          filename = f"pred_{int(video_num[idx])}.png"
          save_image(last_frame[idx], os.path.join(out_folder_name,filename))
          print(f"Saved last frame of video {int(video_num[idx])} to {os.path.join(out_folder_name,filename)}")

        # return real, last_frame, pred, video_num

#predict_one_frame_autoregressive(data_path= , ckpt_path, start, end, output_dir)
predict_one_frame_autoregressive(data_path= "/scratch/ak11089/final-project/raw-data-1/",ckpt_path= "/scratch/ak11089/final-project/next-frame-big-11v-1-cont/logs/checkpoint_27500.pt", start= 15000,end= 15027,output_dir="/scratch/ak11089/final-project//hidden-out/")
