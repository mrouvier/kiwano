#!/bin/bash
#SBATCH --job-name=kw2d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --exclude=apollon,eris,helios
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=kw2d_output.log
#SBATCH --error=kw2d_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_ecapa_tdnn.py --save_path exps/exp3 --feat_type wav2vec2 --feat_dim 768 --n_cpu 10 --batch_size 128 --is_2d

conda deactivate