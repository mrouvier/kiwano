#!/bin/bash
# job name: kf (fbank other) kf_1 (fbank kiwano)
#SBATCH --job-name=fbank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# #SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --constraint=GPURAM_Min_24GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

# python3 utils/train_ecapa_tdnn_2.py --save_path exps/exp1_1  --feat_dim 81 --n_cpu 10 --batch_size 128
# python3 utils/train_ecapa_tdnn.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 10 --batch_size 128
python3 utils/train_ecapa_tdnn.py --save_path exps/exp1_ddp --feat_type fbank --feat_dim 80 --n_cpu 16 --batch_size 512
conda deactivate