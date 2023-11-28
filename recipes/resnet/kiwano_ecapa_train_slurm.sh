#!/bin/bash
#SBATCH --job-name=kiwano_ecapa_train
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=50
#SBATCH --output=kiwano_ecapa_train_output.log
#SBATCH --error=kiwano_ecapa_train_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_ecapa_tdnn.py

conda deactivate
