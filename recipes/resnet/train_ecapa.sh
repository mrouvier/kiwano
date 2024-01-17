#!/bin/bash
#SBATCH --job-name=train_ecapa_kiwano
#SBATCH --partition=gpu
# #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
# #SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
# #SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=5
#SBATCH --output=train_ecapa_kiwano_output.log
#SBATCH --error=train_ecapa_kiwano_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_ecapa_tdnn.py --save_path exps/exp3 --feat_type fbank --feat_dim 80 --n_cpu 5

conda deactivate