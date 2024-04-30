#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=Kiwano
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH -A hho@v100
#SBATCH -C v100-32g


export OMP_NUM_THREADS=10
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1

module purge
#module load pytorch-gpu/py3/1.13.0
module load pytorch-gpu/py3/1.7.1+nccl-2.8.3-1

srun python3  utils/train_resnet_ddp_v2h_perturb_speed_jeffrey.py  data/voxceleb2/ exp/resnet_v2h_perturb_speed_jeffrey/
