#!/bin/bash
#SBATCH --job-name=eval_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=20
#SBATCH --output=eval_output.log
#SBATCH --error=eval_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3  trainECAPAModel.py --eval --initial_model exps/exp4/model/model_0001.model  --feat_type wav2vec2 --feat_dim 768
# python3 trainECAPAModel.py --eval --initial_model exps/pretrain.model --feat_type fbank --feat_dim 80

conda deactivate