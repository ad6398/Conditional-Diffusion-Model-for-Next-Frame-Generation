# Deep-Learning-Project-Fall-23

## Training Diffusion model

1. to train diffusion model for next frame prediction, change the following variable in  `Deep-Learning-Project-Fall-23/src/mcvd/train.sh` file.
	`data`: should contain train, validation and hidden data folders
	`exp`: output folder to save result
	change `CUDA_VISIBLE_DEVICES=0` argument if training on multi GPU and change batch size in file pointed by `config`.

	run train.sh to start training.

	This will generate checkpoint in your folder which you can use for inference in step 3. 

2. To train Unet model, run `python Deep-Learning-Project-Fall-23/src/Unet/train.py /path/to/data/folder/containing/train/val/folder`. This will generate checkpoint which can be used in step 4. example: `python Deep-Learning-Project-Fall-23/src/Unet/train.py ../raw-data-1/dataset/`


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





