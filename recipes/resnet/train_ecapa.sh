#!/bin/bash
# job name: kf (fbank other) kf_1 (fbank kiwano)
#SBATCH --job-name=kf_1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_ecapa_tdnn_2.py --save_path exps/exp1_1  --feat_dim 81 --n_cpu 10 --batch_size 128
# python3 utils/train_ecapa_tdnn.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128

conda deactivate