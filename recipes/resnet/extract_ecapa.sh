!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --job-name=ke
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=5
#SBATCH --time=3:00:00
#SBATCH --array=0-10


dir=exp/resnet/
model=$1

mkdir -p ${dir}/cnceleb2/

python3 utils/extract_resnet.py --world_size=$SLURM_ARRAY_TASK_COUNT  --rank=$SLURM_ARRAY_TASK_ID data/cnceleb2/ ${model} pkl:${dir}/cnceleb2/xvector.$SLURM_ARRAY_TASK_ID.pkl