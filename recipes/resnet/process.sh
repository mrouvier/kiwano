#!/bin/bash
#SBATCH --job-name=process_kiwano
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=128GB
#SBATCH --output=process_kiwano_output.log
#SBATCH --error=process_kiwano_error.log


source /etc/profile.d/conda.sh
conda activate kiwano

# rm -rf db
# rm -rf data

#Prepare Voxceleb1
#python3 local/download_voxceleb1.py db/voxceleb1/
#python3 local/prepare_voxceleb1.py db/voxceleb1/ data/voxceleb1/

#Prepare Voxceleb2
#python3 local/download_voxceleb2.py db/voxceleb2/
#python3 local/prepare_voxceleb2.py db/voxceleb2/ data/voxceleb2/

#Prepare MUSAN
#python3 local/download_musan.py db/musan/
#python3 local/prepare_musan.py db/musan/ data/musan/

#Prepare RIRS NOISES
#python3 local/download_rirs_noises.py db/rirs_noises/
#python3 local/prepare_rirs_noises.py db/rirs_noises/ data/rirs_noises

cp -rf data/voxceleb2/wav db/voxceleb2/

conda deactivate