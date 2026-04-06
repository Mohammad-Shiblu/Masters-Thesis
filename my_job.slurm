#!/bin/bash -l
#SBATCH --job-name=Casacded_residual
#SBATCH --partition=work
#SBATCH --gres=gpu:1                 
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwi5/iwi5240h/Masters-Thesis/log/%x_%j.log
#SBATCH --error=/home/woody/iwi5/iwi5240h/Masters-Thesis/log/%x_%j.log


module load python/3.12-conda

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /home/woody/iwi5/iwi5240h/conda_envs/thesis

echo "Host: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "which python: $(which python)"
python -V || { echo "Python not found"; exit 2; }

python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"

cd /home/woody/iwi5/iwi5240h/Masters-Thesis
python main.py
