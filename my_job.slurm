#!/bin/bash
#SBATCH --job-name=UNET_lpips(MSE_loss)
#SBATCH --output=/dev/null
#SBATCH --error=error_file.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00  # 1 hour

output_dir="/home/woody/iwi5/iwi5240h/Masters-Thesis/log"

log_file="$output_dir/$SLURM_JOB_NAME"_"$SLURM_JOB_ID.log"

exec &> $log_file

# Run your Python model
python train.py
