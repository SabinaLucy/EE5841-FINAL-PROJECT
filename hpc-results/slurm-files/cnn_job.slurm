#!/bin/bash
#SBATCH --job-name=cnn_gpu
#SBATCH --output=cnn_out.txt
#SBATCH --error=cnn_err.txt
#SBATCH --time=01:00:00
#SBATCH --partition=mrigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source /home/nehavila/miniconda3/etc/profile.d/conda.sh
conda activate cs-ee5841
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd /home/nehavila/assignments
python cnn-model.py
