
export user="ak11089"

#export main_folder="/home/${user}/scratch/dl-project"
# export exp_folder=${main_folder}/checkpoints
export config="semantic_seg_DDPM_big5"
export data="${DL_PROJ}raw-data-1/dataset"

# Video prediction non-spade
export exp="${DL_PROJ}semantic_pred_5v5-debug-1"
export config_mod=""


CUDA_VISIBLE_DEVICES=0 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
