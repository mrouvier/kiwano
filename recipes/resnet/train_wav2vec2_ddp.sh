#!/bin/bash
# job names: kw2v2b_ddp (base), kw2v2l_ddp (large), kw2v2r_ddp (robust), kw2v2s_ddp (self)
#SBATCH --job-name=kw2v2b_ddp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

# python3  utils/train_ecapa_tdnn_ddp.py --eval --initial_model exps/exp2_1_ddp/model/model_0001.model  --feat_type wav2vec2  --n_cpu 10 --model_name facebook/wav2vec2-large-960h
# python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp2_1_ddp --feat_type wav2vec2  --n_cpu 10 --batch_size 128 --model_name facebook/wav2vec2-large-960h
python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp_wav2vec2_base_960h --feat_type wav2vec2  --n_cpu 10 --batch_size 128 --model_name facebook/wav2vec2-base-960h
conda deactivate