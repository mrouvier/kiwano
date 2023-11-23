#!/bin/bash
#SBATCH --job-name=kiwano
#SBATCH --partition=gpu
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=15
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=kiwano_output.log
#SBATCH --error=kiwano_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

export OMP_NUM_THREADS=15
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1


python3 utils/train_wav2vec2.py
# python3 -m pdb utils/train_wav2vec2.py

conda deactivate
