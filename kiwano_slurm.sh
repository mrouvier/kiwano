#!/bin/bash
#SBATCH --job-name=kiwano
#SBATCH --partition=gpu
#SBATCH --time=0-00:00
#SBATCH --output=kiwano_output.log
#SBATCH --error=kiwano_error.log
#SBATCH --mail-type=BEGIN, END
#SBATCH --mail-user=lamahmichelmarie@gmail.com

source /etc/profile.d/conda.sh
conda activate kiwano

python3 train_wav2vec2.py

conda deactivate