#!/bin/bash
#SBATCH --job-name=kw2d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Max_24GB
#SBATCH --exclude=alpos
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=kw2d_output.log
#SBATCH --error=kw2d_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp3_3 --feat_type wav2vec2 --feat_dim 768 --n_cpu 10 --batch_size 128 --is_2d --model_name facebook/wav2vec2-large-960h-lv60-self

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp3_2 --feat_type wav2vec2 --feat_dim 768 --n_cpu 10 --batch_size 128 --is_2d --model_name facebook/wav2vec2-large-robust-ft-libri-960h

python3 utils/train_ecapa_tdnn.py --save_path exps/exp3_1 --feat_type wav2vec2 --feat_dim 768 --n_cpu 10 --batch_size 128 --is_2d --model_name facebook/wav2vec2-large-960h

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp3 --feat_type wav2vec2 --feat_dim 768 --n_cpu 10 --batch_size 128 --is_2d

conda deactivate