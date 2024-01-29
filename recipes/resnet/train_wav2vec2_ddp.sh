#!/bin/bash
# job names: kw_ddp (base), kw_1_ddp (large), kw_2_ddp (robust), kw_3_ddp (self)
#SBATCH --job-name=kw_1_ddp
# #SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPURAM_Max_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=10
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

python3 utils/train_ecapa_tdnn_ddp.py --save_path exps/exp2_1_ddp --feat_type wav2vec2  --n_cpu 10 --batch_size 64 --model_name facebook/wav2vec2-large-960h

conda deactivate