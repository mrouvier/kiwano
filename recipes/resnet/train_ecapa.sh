#!/bin/bash
#SBATCH --job-name=kf
#SBATCH --partition=gpu
# #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
# #SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
# #SBATCH --gres=gpu:rtx_3090:1
# #SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --output=kf_output.log
#SBATCH --error=kf_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_ecapa_tdnn.py --save_path exps/exp1 --feat_type fbank --feat_dim 80 --n_cpu 4 --batch_size 256

conda deactivate