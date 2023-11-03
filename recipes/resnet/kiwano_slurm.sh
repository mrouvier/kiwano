#!/bin/bash
#SBATCH --job-name=kiwano
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --output=kiwano_output.log
#SBATCH --error=kiwano_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_wav2vec2.py

conda deactivate