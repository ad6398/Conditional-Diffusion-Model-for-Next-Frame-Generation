import glob, os
import mediapy as media
import torch
from torch.utils.data import DataLoader

from load_model_from_ckpt import load_model, get_sampler, init_samples
from datasets import get_dataset, data_transform, inverse_data_transform
from runners.ncsn_runner import conditioning_fn


import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

def save_last_frame(pred, video_num):
    # Assuming 'pred' is a PyTorch tensor of shape (batch_size, num_frames, image_width, image_height)
    # Assuming 'video_num' is a PyTorch tensor of shape (batch_size, 1)

    batch_size, num_frames, image_width, image_height = pred.shape

    # Extract the last frame for each batch
    last_frames = pred[:, -1, :, :].view(batch_size, 1, image_width, image_height)

    # Convert last_frames tensor to images
    image_transform = transforms.ToPILImage()
    images = [image_transform(last_frames[i]) for i in range(batch_size)]

    # Save images with corresponding video_num suffix
    for i in range(batch_size):
        filename = f"pred_{int(video_num[i])}.png"
        save_image(last_frames[i], filename)
        print(f"Saved last frame of video {int(video_num[i])} to {filename}")



	



def main(data_path, ckpt_path):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	scorenet, config = load_model(ckpt_path, device)
	sampler = get_sampler(config)
	dataset, test_dataset = get_dataset(data_path, config, video_frames_pred=config.data.num_frames)

	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                         num_workers=config.data.num_workers, drop_last=True)
	
	for i, test_x, test_y in enumerate(test_loader):
		test_x = data_transform(config, test_x)
		real, cond, cond_mask = conditioning_fn(config, test_x, num_frames_pred=config.data.num_frames,
                                        prob_mask_cond=getattr(config.data, 'prob_mask_cond', 0.0),
                                        prob_mask_future=getattr(config.data, 'prob_mask_future', 0.0))
		
		init = init_samples(len(real), config)
		pred = sampler(init, scorenet, cond=cond, cond_mask=cond_mask, subsample=100, verbose=True)
        save_last_frame(pred, test_y)
      
	


      



	

