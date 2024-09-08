#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=Kiwano
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH -A hho@v100
#SBATCH -C v100-32g


export OMP_NUM_THREADS=10
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1


module purge
module load pytorch-gpu/py3/2.3.0

srun python utils/train_resnet_ddp_wd_sgd_mini.py  data/ASVSpoof5/ exp/resnet_ddp_adamw_batch256_mini/
