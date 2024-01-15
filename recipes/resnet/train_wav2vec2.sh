#!/bin/bash
#SBATCH --job-name=train_wav2vec2_kiwano
#SBATCH --partition=gpu
# #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --time=7-00:00:00
# #SBATCH --mem=128GB
#SBATCH --cpus-per-task=5
#SBATCH --output=train_wav2vec2_kiwano_output.log
#SBATCH --error=train_wav2vec2_kiwano_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_ecapa_tdnn.py --save_path exps/exp4 --feat_type wav2vec2 --feat_dim 768 --n_cpu 5

conda deactivate