
export user="ak11089"

export main_folder="/home/${user}/scratch/dl-project"
export exp_folder=${main_folder}/checkpoints
export config="semantic_seg_DDPM_big5"
export data="${data_folder}"

# Video prediction non-spade
export exp="semantic_pred_5v5"
export config_mod="training.snapshot_freq=50000 sampling.subsample=100 sampling.clip_before=True sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore model.num_res_blocks=2"


CUDA_VISIBLE_DEVICES=0 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}