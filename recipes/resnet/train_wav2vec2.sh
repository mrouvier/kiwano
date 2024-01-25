#!/bin/bash
# job names: kw (base), kw_1 (large), kw_2 (robust), kw_3 (self)
#SBATCH --job-name=kw_1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_Max_40GB
#SBATCH --exclude=alpos
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp2_3 --feat_type wav2vec2  --n_cpu 10 --batch_size 128 --model_name facebook/wav2vec2-large-960h-lv60-self

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp2_2 --feat_type wav2vec2  --n_cpu 10 --batch_size 128 --model_name facebook/wav2vec2-large-robust-ft-libri-960h

python3 utils/train_ecapa_tdnn.py --save_path exps/exp2_1 --feat_type wav2vec2  --n_cpu 10 --batch_size 128 --model_name facebook/wav2vec2-large-960h

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp2 --feat_type wav2vec2 --n_cpu 10 --batch_size 128

conda deactivate