#!/bin/bash
# job name: kf_ddp (fbank other) kf_1_ddp (fbank kiwano)
#SBATCH --job-name=fbank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
# #SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

# python3  utils/train_ecapa_tdnn_ddp.py --eval --initial_model exps/exp1_ddp/model/model_0006.model  --feat_type fbank --feat_dim 80 --n_cpu 10
# python3  utils/train_ecapa_tdnn_2_ddp.py --eval --initial_model exps/exp1_1_ddp/model/model_0006.model  --feat_dim 81 --n_cpu 10
# python3 utils/train_ecapa_tdnn_2_ddp.py --save_path exps/exp1_1_ddp  --feat_dim 81 --n_cpu 10 --batch_size 128 --master_port 54321
python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp1_ddp --feat_type fbank --feat_dim 80 --n_cpu 16 --batch_size 128 --master_port 54322

conda deactivate