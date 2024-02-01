#!/bin/bash
# job names: bwlmb_ddp (base),  bwlmbcv_ddp (base cv), bwlmbp_ddp (base plus), bwlmbpcv_ddp (base plus cv), bwlml_ddp (large),
#SBATCH --job-name=bwlmb_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Max_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano


# python3 train_ecapa_tdnn.py --save_path exps/exp_wavlm_large_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-large

# python3 train_ecapa_tdnn.py --save_path exps/exp_wavlm_base_plus_sv_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base-plus-sv

# python3 train_ecapa_tdnn.py --save_path exps/exp_wavlm_base_plus_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base-plus

# python3 train_ecapa_tdnn.py --save_path exps/exp_wavlm_base_sv_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base-sv

python3 train_ecapa_tdnn.py --save_path exps/exp_wavlm_base_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base

# python3 -m pdb train_ecapa_tdnn.py --save_path exps/exp_wavlm_base_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base

conda deactivate