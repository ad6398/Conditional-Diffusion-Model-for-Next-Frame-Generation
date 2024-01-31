# Conditional Diffusion Model for Next Frame Generation
**Deep Learning, taught by Prof Yann LeCun & Alfredo Canziani** - final exam competition.

## Problem Statement

Given 11 frames of videos, where simple 3D shapes interact with each other according to basic physics principles, predict 22nd frame along with its semantic segmentation.

### Dataset Description

The dataset features synthetic videos with simple 3D shapes that interact with each other according to basic physics principles. Objects in videos have three shapes (cube, sphere, and cylinder), two materials (metal and rubber), and eight colors (gray, red, blue, green, brown, cyan, purple, and yellow). In each video, there is no identical objects, such that each combination of the three attributes uniquely identifies one object.

For unlabeled, training and validation set, all 22 frames are there for each video. For hidden set only the first 11 frames of each video.

For training set and validation set, the full semantic segmentation mask for each frame are given.

### Task Description

The task on hidden set is to use the first 11 frames to generate the semantic segmentation mask of the last frame (the 22nd frame). The performance is evaluated by calculate the IOU/Jaccard between the ground truth and the generated mask.

## Solution
Our solution is divided into two separate problems: Next Frame Prediction and Semantic Segmentation. Semantic segmentation was an easy task that we solved using a Unet model (pixel classification). We focus on Next Frame generation.


### Next Frame Generation via Conditional Diffusion
We condition a Diffusion model on the concatenation of the previous frames, in sequence, to generate the next frame. Then, we autoregressively/directly generate the 22nd frame. Below is architecture that we used.

<img width="985" alt="Screenshot 2024-01-31 at 9 05 15 AM" src="https://github.com/ad6398/Conditional-Diffusion-Model-for-Next-Frame-Generation/assets/38162294/868e94c2-62ae-4e18-a21e-855bc87892e4">


### Experiments and Results

Our Unet yielded a Jaccard score of 96.9 on the validation data for the semantic segmentation task. For the next frame generation, we conducted the following experiments, and for the final metric, we calculated the Jaccard score of the semantic segmentation for the generated 22nd frame.

<img width="900" alt="Screenshot 2024-01-31 at 9 24 44 AM" src="https://github.com/ad6398/Conditional-Diffusion-Model-for-Next-Frame-Generation/assets/38162294/5fb24f50-0c35-4556-a640-e1f02fed61e9">

Our model was trained on 13,000 unlabeled partition of given dataset. Each frame is resized from 160 × 240 into 128 × 128.

* The first training experiment involved predicting the final 22nd frame directly by conditioning on the first 11 frames. This resulted in a Jaccard Index of 0.198 on validation set. However, it was observed that the loss seemed to stagnate after 30 epochs. This experiment was treated as the baseline.
  
* The second training experiment involved predicting six future frames i.e., 12th, 14th, 16th, 18th, 20th and 22nd frame, conditioned on five past frames i.e., 2nd, 4th, 6th, 8th and 10th. This resulted in a Jaccard Index of 0.136 on validation set. However, it was observed that loss seemed not to converge.
  
* The final training experiment involved autoregressively predicting all future frames conditioned on past 11 frames i.e., predicting the 12th frame conditioned on frames 1-11, then predicting the 13th frame conditoned on frames 2-12 utilzing the predicted 12th frame from the previous step. This resulted in a Jaccard Index of 0.308 on validation set, giving us the best result.


**Plots for our training and validation losses**

<img width="610" alt="Screenshot 2024-01-31 at 9 38 06 AM" src="https://github.com/ad6398/Conditional-Diffusion-Model-for-Next-Frame-Generation/assets/38162294/a2bcf5a9-c161-43ca-8eb4-2b500efbc6df">



**Glimpse of 11 input frames, original 12th frame and generated 12th frame**


<img width="704" alt="Screenshot 2024-01-31 at 10 00 21 AM" src="https://github.com/ad6398/Conditional-Diffusion-Model-for-Next-Frame-Generation/assets/38162294/43acab6e-830a-4db1-800a-609fc316374e">



## Replicating our experiments 

### Training

1. to train diffusion model for next frame prediction, change the following variable in  `Deep-Learning-Project-Fall-23/src/mcvd/train.sh` file.
	`data`: should contain train, validation and hidden data folders
	`exp`: output folder to save result
	change `CUDA_VISIBLE_DEVICES=0` argument if training on multi GPU and change batch size in file pointed by `config`.

	run train.sh to start training.

	This will generate checkpoint in your folder which you can use for inference in step 3. 

2. To train Unet model, run `python Deep-Learning-Project-Fall-23/src/Unet/train.py /path/to/data/folder/containing/train/val/folder`. This will generate checkpoint which can be used in step 4. example: `python Deep-Learning-Project-Fall-23/src/Unet/train.py ../raw-data-1/dataset/`

### Inference

3. Post training Unet, we need to do inference on hidden set to predict 22nd frame on hidden data. Run `Deep-Learning-Project-Fall-23/src/mcvd/test_diffusion_hidden.py DATA_PATH CKPT_PATH OUT_DIR` where:

	`DATA_PATH`: folder containing hidden folder
	
	`CKPT_PATH`: path to .pt file generated in step 1, which will be used.
	
 	`OUT_DIR`: path to dir where  output image will be stored.

	example: `python Deep-Learning-Project-Fall-23/src/mcvd/test_diffusion_hidden.py  raw-data-1/ /scratch/ak11089/final-project/next-frame-big-11v-1-cont/logs/checkpoint_27500.pt /scratch/ak11089/final-project/`

4. Now we need to use our trained Unet model to generate segmentation of predicted frames in step 3. To do so run `python Deep-Learning-Project-Fall-23/src/run_segmentation.py UNET_PATH, PRED_PATH` where:

	`UNET_PATH`: path to unet checkpoint created in step 2
	
 	`PRED_PATH`: path of the folder created inside `OUT_DIR` in step.

 	`example: python Deep-Learning-Project-Fall-23/src/run_segmentation.py  Unet/unet-test/unet1.pt final_pred_hidden/all/`
This step will generate .pt file containing our prediction. 





