#!/bin/bash
#SBATCH --job-name=kiwano_ecapa_eval
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=50
#SBATCH --output=kiwano_ecapa_eval_output.log
#SBATCH --error=kiwano_ecapa_eval_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_ecapa_tdnn.py --eval --score_path exps/pretrain/kiwano_score.txt --model_path exps/pretrain/pretrain.model

conda deactivate
