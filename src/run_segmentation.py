import os
import torch
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from Unet.train import SegData,EncodingBlock,unet_model
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torchmetrics
from torchvision import transforms
import re, sys


# change thsi to your unet path



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PredictedImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = os.listdir(directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        #print(self.images[idx])
        pattern = re.compile(r'pred_(\d+)$')
        match = pattern.search(self.images[idx])
        #print(self.images[idx][-9:-4])
        video_number = int(self.images[idx][-9:-4])
        #print(video_number)
        
        if self.transform:
            image = self.transform(image)
        
        return image, video_number

class UnNormalize(object):
    def __call__(self, tensor):
        return tensor * 255


def load_validation_masks(val_dir, start_index, end_index, new_size=(128, 128)):
    validation_masks = []
    resize_transform = transforms.Resize(new_size)
    for i in range(start_index, end_index):
        mask_path = f'{val_dir}{i}/mask.npy'
        mask = np.load(mask_path)
        mask_image = Image.fromarray(mask[21])  # Convert the 22nd mask to an image
        #mask_resized = resize_transform(mask_image)  # Resize the mask
        validation_masks.append(np.array(mask_image))
    return validation_masks




jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)

# softmax = torch.nn.Softmax(dim=1)
jaccard_scores = []
total_dp = 0

def validate(val_path, ckpt):
    result = predict(ckpt, val_path)
    # Convert numpy array to torch tensor for validation mask
    val_masks = load_validation_masks('/scratch/ak11089/final-project//raw-data-1/dataset/val/video_', 1000, 2000, new_size=(128, 128))
    preds = []
    masks = []
    tt = 0
    avg_jac = 0.0
    for vn in result:
            
        val_mask_tensor = torch.from_numpy(val_masks[vn-1000]).unsqueeze(0)
        masks.append(val_mask_tensor)
        preds.append(result[vn].to("cpu"))
        avg_jac += jaccard(result[vn].to("cpu"), val_mask_tensor)
        #print(val_mask_tensor.shape, result[vn].shape)
        tt += 1
    # Compute Jaccard Index for the current pair of masks
    #print(preds)
    print("Sujana's Jaccard", avg_jac /tt)
    preds = torch.concat(preds, dim = 0)
    masks = torch.concat(masks, dim = 0)
    print("final pred shape",preds.shape)
    score = jaccard(preds, masks)
    print("jacc score", score)
  
def predict(ckpt, data_path):
    #DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = unet_model().to(DEVICE)
    m = torch.load(ckpt).state_dict()
    model.load_state_dict(m)
    model.eval()
    pred_dataset = PredictedImageDataset(directory=data_path, transform = transforms.Compose([transforms.Resize((160,240)),transforms.ToTensor(),UnNormalize()]))
    #pred_dataset = PredictedImageDataset(directory=data_path, transform = transforms.Compose([transforms.Resize((160,240)),transforms.ToTensor()]))
    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
    results = {}
    for i, (image, video_number) in enumerate(tqdm(pred_loader, desc="Generating Masks")):
        image = image.to(DEVICE)
        with torch.no_grad():
            output = model(image)
            preds = torch.argmax(output, axis=1)
        #print(preds.shape)
        #print(video_number)
        for j in range(video_number.shape[0]):
            results[video_number[j].item()] = preds[j].unsqueeze(0)



    return results

def save_results_pt(result):
    video_nums = list(result.keys())
    video_nums.sort()
    start = 15000
    idx = 0
    res_list = []
    for vn in video_nums:
        #print(vn)
        assert vn == start, f"{start} is not present in your prediction"
        pred = result[vn]
        res_list.append(pred.to("cpu"))
        start += 1

    #print("total dp", start)
    assert start == 17000
    result_tensor = torch.concat(res_list, dim = 0)
    print("tensor final shape",result_tensor.shape)
    torch.save(result_tensor, "final_leaderboard_team_27.pt")


if __name__ == "__main__":
    
    unet_path = sys.argv[1]
    diffusion_pred = sys.argv[2]
    r = predict(unet_path,diffusion_pred )
    save_results_pt(r)

# r = predict(unet_model_saved_path,"/scratch/ak11089/final-project//final_pred_hidden/all/" )
# save_results_pt(r)

# validate("/scratch/ak11089/final-project/val_ad", unet_model_saved_path)
