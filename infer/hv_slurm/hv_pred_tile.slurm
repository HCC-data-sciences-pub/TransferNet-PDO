#!/bin/bash
#SBATCH --job-name=hovernet
#SBATCH -N 1                              
#SBATCH --time=3:00:00                   
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:4
#SBATCH --mem=128g

# load module needed
module load cuda/11.1.1
module load rclone/1.53.2

# setup CUDA
CUDA_LAUNCH_BLOCKING=1
echo "GPU nums $CUDA_VISIBLE_DEVICES"

# copy files to slurm scratch
echo "Copying file to slurm scratch"
SRC="/path/to/input_data/directory/"              
mkdir $SLURM_SCRATCH/data
rclone copy $SRC $SLURM_SCRATCH/data  --multi-thread-streams=8
echo "Copy complete"

# start predicting cell types
echo "Begin Processing!"

/path/to/hover-net_env/bin/python     /path/to/run_infer.py       \
--nr_types=3                                                      \
--type_info_path="/path/to/type_info.json"                        \
--batch_size=32                                                   \
--model_mode=fast                                                 \
--model_path="/path/to/model_checkpoint/model.tar"                \
tile \
--input_dir=$SLURM_SCRATCH/data                                   \
--output_dir="/path/to/output_dir/"   
echo "Finished"
