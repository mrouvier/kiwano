#!/bin/bash
#SBATCH --job-name=kf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --exclude=apollon,eris,helios
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=kf_output.log
#SBATCH --error=kf_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_ecapa_tdnn.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128

conda deactivate