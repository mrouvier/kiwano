#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=Kiwano
#SBATCH --cpus-per-task=10
#SBATCH --time=0:50:00
#SBATCH -A rmw@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --array=0-20
#SBATCH -C v100-32g

#module load pytorch-gpu/py3/1.7.1+nccl-2.8.3-1

export OMP_NUM_THREADS=10

module purge
module load pytorch-gpu/py3/2.2.0

dir=$1

mkdir -p ${dir}/asvspoof.${2}/
mkdir -p ${dir}/cfad_noisy.${2}/
mkdir -p ${dir}/cfad_codec.${2}/
mkdir -p ${dir}/latin_american.${2}/
mkdir -p ${dir}/itw.${2}/
#mkdir -p ${dir}/defi.${2}/
#mkdir -p ${dir}/defi_taskb.${2}/



python3 utils/extract_mhfa.py --world_size=$SLURM_ARRAY_TASK_COUNT  --rank=$SLURM_ARRAY_TASK_ID data/latin_american/ ${dir}/model${2}.ckpt pkl:${dir}/latin_american.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl

python3 utils/extract_mhfa.py --world_size=$SLURM_ARRAY_TASK_COUNT  --rank=$SLURM_ARRAY_TASK_ID data/ASVSpoof5_dev/ ${dir}/model${2}.ckpt pkl:${dir}/asvspoof.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl

python3 utils/extract_mhfa.py --world_size=$SLURM_ARRAY_TASK_COUNT  --rank=$SLURM_ARRAY_TASK_ID data/cfad_noisy_version/ ${dir}/model${2}.ckpt pkl:${dir}/cfad_noisy.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl

python3 utils/extract_mhfa.py --world_size=$SLURM_ARRAY_TASK_COUNT  --rank=$SLURM_ARRAY_TASK_ID data/cfad_codec_version/ ${dir}/model${2}.ckpt pkl:${dir}/cfad_codec.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl

python3 utils/extract_mhfa.py --world_size=$SLURM_ARRAY_TASK_COUNT  --rank=$SLURM_ARRAY_TASK_ID data/itw/ ${dir}/model${2}.ckpt pkl:${dir}/itw.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl

#python3 utils/extract_mhfa.py --world_size=$SLURM_ARRAY_TASK_COUNT  --rank=$SLURM_ARRAY_TASK_ID data/defi/ ${dir}/model${2}.ckpt pkl:${dir}/defi.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl

#python3 utils/extract_mhfa.py --world_size=$SLURM_ARRAY_TASK_COUNT  --rank=$SLURM_ARRAY_TASK_ID data/defi_taskb/ ${dir}/model${2}.ckpt pkl:${dir}/defi_taskb.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl
