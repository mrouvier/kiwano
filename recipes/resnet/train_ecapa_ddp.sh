#!/bin/bash
#SBATCH --job-name=kf_1_ddp
# #SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_Max_12GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 utils/train_ecapa_tdnn_2_ddp.py --save_path exps/exp1_1_ddp  --feat_dim 81 --n_cpu 10 --batch_size 128

conda deactivate