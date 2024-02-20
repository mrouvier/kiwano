#!/bin/bash
# job names: bwlmb (base),  bwlmbcv (base cv), bwlmbp (base plus), bwlmbpcv (base plus cv), bwlml (large),
#SBATCH --job-name=bwlmb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB&GPURAM_Max_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano


# python3 utils/train_ecapa_tdnn.py --save_path exps/exp_wavlm_large --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-large

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp_wavlm_base_plus_sv --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base-plus-sv

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp_wavlm_base_plus --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base-plus

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp_wavlm_base_sv --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base-sv

python3 utils/train_ecapa_tdnn.py --save_path exps/exp_wavlm_base --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base

conda deactivate