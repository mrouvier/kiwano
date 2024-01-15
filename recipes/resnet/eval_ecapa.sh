#!/bin/bash
#SBATCH --job-name=eval_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=7-00:00:00
# #SBATCH --mem=128GB
#SBATCH --cpus-per-task=5
#SBATCH --output=eval_output.log
#SBATCH --error=eval_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

# python3  utils/train_ecapa_tdnn.py --eval --initial_model exps/exp4/model/model_0004.model  --feat_type wav2vec2 --feat_dim 768 --n_cpu 5
python3 utils/train_ecapa_tdnn.py --eval --initial_model exps/exp3/model_0013.model --feat_type fbank --feat_dim 80 --n_cpu 5

conda deactivate