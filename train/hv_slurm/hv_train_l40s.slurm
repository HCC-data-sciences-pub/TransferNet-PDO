#!/bin/bash
#SBATCH --job-name=hv_train
#SBATCH -N 1                             # Ensure all cores on one machine
#SBATCH --time=12:00:00                  # HH:MM:SS
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:4                     # request 4  gpu card [0,1,2,3]
#SBATCH --mem=256g                       # memory of cpu (gpu machine also has cpu)

module load cuda/11.1.1
CUDA_LAUNCH_BLOCKING=1

echo "Begin Processing!"

/path/to/hover-net_env/bin/python /path/to/run_train.py \
--gpu='0,1,2,3' 

echo "Finished"
