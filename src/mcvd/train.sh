
export user="ak11089"

#export main_folder="/home/${user}/scratch/dl-project"
# export exp_folder=${main_folder}/checkpoints
export config="next_frame_DDPM_big5v6"
export data="${DL_PROJ}raw-data-1/dataset"

export exp="${DL_PROJ}next-frame-big-5v-6"
export config_mod=""


CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --resume_training --config_mod ${config_mod}
#CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/${config}.yml --data_path ${data} --exp ${exp} --ni --config_mod ${config_mod}
