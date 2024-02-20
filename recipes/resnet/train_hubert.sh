#!/bin/bash
# job name: khl_ll60k (hubert-large-ll60k)
# job name: khxl_ll60k (hubert-xlarge-ll60k)
# job name: khl_ls960_ft (hubert-large-ls960-ft)
# job name: khxl_ls960_ft(hubert-xlarge-ls960-ft)
# job name: khb_ls960 (hubert-base-ls960)


#SBATCH --job-name=khb_ls960
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB&GPURAM_Max_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano


# python3 utils/train_ecapa_tdnn.py --save_path exps/exp_hubert_large_ll60k --feat_type hubert --n_cpu 10 --batch_size 128 --model_name facebook/hubert-large-ll60k

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp_hubert_xlarge_ll60k --feat_type hubert --n_cpu 10 --batch_size 128 --model_name facebook/hubert-xlarge-ll60k

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp_hubert_large_ls960_ft --feat_type hubert --n_cpu 10 --batch_size 128 --model_name facebook/hubert-large-ls960-ft

# python3 utils/train_ecapa_tdnn.py --save_path exps/exp_hubert_xlarge_ls960_ft --feat_type hubert --n_cpu 10 --batch_size 128 --model_name  facebook/hubert-xlarge-ls960-ft

python3  utils/train_ecapa_tdnn.py --save_path exps/exp_hubert_base_ls960 --feat_type hubert --n_cpu 10 --batch_size 64 --model_name facebook/hubert-base-ls960

conda deactivate