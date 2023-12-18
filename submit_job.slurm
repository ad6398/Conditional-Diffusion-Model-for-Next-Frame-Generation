#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_001-2023fa-27

## change the partition number to use different number of GPUs
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

##SBATCH --partition=n1c24m128-v100-4
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=24

#SBATCH --time=07:00:00
#SBATCH --output=hidden_infer_%j.out
#SBATCH --error=hidden_infer_%j.err
#SBATCH --exclusive
#SBATCH --requeue


singularity exec --bind /scratch/ak11089 --nv --overlay /scratch/ak11089/final-project/overlay-50G-10M.ext3:ro /share/apps/images/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/activate_conda.sh
cd /scratch/ak11089/final-project/Deep-Learning-Project-Fall-23/src/mcvd

sh src/mcvd/train.sh
"
##CUDA_VISIBLE_DEVICES=0 python test_diffusion.py
