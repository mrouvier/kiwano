#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=Kiwano
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH -A bbo@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --array=0-80
#SBATCH -C v100-32g

export OMP_NUM_THREADS=10

module purge
module load pytorch-gpu/py3/2.2.0

dir=$1

mkdir -p ${dir}/voxceleb1.${2}/
python3 utils/extract_redimnet.py --world_size=$SLURM_ARRAY_TASK_COUNT --rank=$SLURM_ARRAY_TASK_ID data/voxceleb1/ ${dir}/model${2}.ckpt pkl:${dir}/voxceleb1.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl


mkdir -p ${dir}/dipco.${2}/
python3 utils/extract_redimnet.py --world_size=$SLURM_ARRAY_TASK_COUNT --rank=$SLURM_ARRAY_TASK_ID data/dipco/ ${dir}/model${2}.ckpt pkl:${dir}/dipco.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl



mkdir -p ${dir}/commonbench.${2}/
python3 utils/extract_redimnet.py --world_size=$SLURM_ARRAY_TASK_COUNT --rank=$SLURM_ARRAY_TASK_ID data/commonbench/ ${dir}/model${2}.ckpt pkl:${dir}/commonbench.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl



mkdir -p ${dir}/cnceleb.${2}/
python3 utils/extract_redimnet.py --world_size=$SLURM_ARRAY_TASK_COUNT --rank=$SLURM_ARRAY_TASK_ID data/cnceleb/ ${dir}/model${2}.ckpt pkl:${dir}/cnceleb.${2}/xvector.$SLURM_ARRAY_TASK_ID.pkl
