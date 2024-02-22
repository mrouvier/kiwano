#!/bin/bash
# job name: khl_ll60k_ddp (hubert-large-ll60k)
# job name: khxl_ll60k_ddp (hubert-xlarge-ll60k)
# job name: khl_ls960_ft_ddp (hubert-large-ls960-ft)
# job name: khxl_ls960_ft_ddp (hubert-xlarge-ls960-ft)
# job name: khb_ls960_ddp (hubert-base-ls960)


#SBATCH --job-name=khxl_ls960_ft_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano


# python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_hubert_large_ll60k_ddp --feat_type hubert --n_cpu 10 --batch_size 128 --model_name facebook/hubert-large-ll60k

# python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_hubert_xlarge_ll60k_ddp --feat_type hubert --n_cpu 10 --batch_size 128 --model_name facebook/hubert-xlarge-ll60k

# python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_hubert_large_ls960_ft_ddp --feat_type hubert --n_cpu 10 --batch_size 128 --model_name facebook/hubert-large-ls960-ft

python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_hubert_xlarge_ls960_ft_ddp --feat_type hubert --n_cpu 10 --batch_size 128 --model_name  facebook/hubert-xlarge-ls960-ft

# python3  utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_hubert_base_ls960_ddp --feat_type hubert --n_cpu 10 --batch_size 128 --model_name facebook/hubert-base-ls960

conda deactivate