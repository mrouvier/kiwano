#!/bin/bash
#SBATCH --job-name=ke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --exclude=apollon,eris
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

# python3 -m pdb utils/train_ecapa_tdnn_2.py --eval --initial_model exps/exp1_1/model/model_0001.model  --feat_dim 81 --n_cpu 0

python3  utils/train_ecapa_tdnn_2.py --eval --initial_model exps/exp1_1/model/model_0001.model  --feat_dim 81 --n_cpu 10

# python3  utils/train_ecapa_tdnn.py --eval --initial_model exps/exp2/model/model_0017.model  --feat_type wav2vec2 --feat_dim 768 --n_cpu 5
# python3 utils/train_ecapa_tdnn.py --eval --initial_model exps/exp1/model/model_0027.model --feat_type fbank --feat_dim 80 --n_cpu 4
# python3 utils/train_ecapa_tdnn.py --eval --initial_model exps/pretrain.model --feat_type fbank --feat_dim 80 --n_cpu 5


conda deactivate