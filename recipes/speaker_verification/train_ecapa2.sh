#!/bin/bash
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=Kiwano
#SBATCH --cpus-per-task=15


export OMP_NUM_THREADS=10

module purge
module load pytorch-gpu/py3/2.3.0

srun python3 utils/train_ecapa2.py data/voxceleb2/ exp/ecapa2/
