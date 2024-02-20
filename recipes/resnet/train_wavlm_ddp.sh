#!/bin/bash
# job names: kwlmb_ddp (base),
# kwlmbcv_ddp (base cv),
# kwlmbp_ddp (base plus),
# kwlmbpcv_ddp (base plus cv),
# kwlml_ddp (large)

#SBATCH --job-name=kwlmb_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_Min_12GB&GPURAM_Max_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano


# python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_wavlm_large_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-large

# python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_wavlm_base_plus_sv_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base-plus-sv

# python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_wavlm_base_plus_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base-plus

# python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_wavlm_base_sv_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base-sv

python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_wavlm_base_ddp --feat_type wavlm --n_cpu 10 --batch_size 64 --model_name microsoft/wavlm-base

# python3 -m pdb utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_wavlm_base_ddp --feat_type wavlm --n_cpu 10 --batch_size 128 --model_name microsoft/wavlm-base


conda deactivate