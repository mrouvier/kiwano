#!/bin/bash
#SBATCH --job-name=vn_celeb
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=64GB
# #SBATCH --cpus-per-task=20
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


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


# cp -rf data/voxceleb2/wav db/voxceleb2/

# rm -rf data/voxceleb2/wav

#rm -rf db/voxceleb2/wav/*
#rm -rf db/voxceleb2/dev
#cp -rf ./../../../dataset/db/voxceleb2/*  db/voxceleb2/wav/
#rm -rf db/voxceleb2/wav/vox2_meta.csv
#mv db/voxceleb2/wav/train_list.txt db/voxceleb2/


# rm -r db/cn_celeb/*
# python3 local/download_cn_celeb.py db/cn_celeb/
#wget https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz  -P db/cn_celeb/
#wget https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzaa -P db/cn_celeb/
#wget https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzab -P db/cn_celeb/
#wget https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzac -P db/cn_celeb/

# cat  db/cn_celeb/cn-celeb2_v2.* >  db/cn_celeb/cn-celeb2.zip
# unzip db/cn_celeb/cn-celeb2.zip -d db/cn_celeb/

# python3 local/prepare_cn_celeb.py db/cn_celeb/ data/cn_celeb/

# python3 local/extract_vietnam_celeb.py db/vietnam_celeb/
# rm  db/vietnam_celeb/vietnam-celeb.zip
# rm  db/vietnam_celeb/vietnam_celeb.zip
# cat  db/vietnam_celeb/vietnam-celeb-part* >  db/vietnam_celeb/vietnam-celeb.zip
# unzip db/vietnam_celeb/vietnam-celeb.zip -d db/vietnam_celeb/
python3 local/prepare_vietnam_celeb.py db/vietnam_celeb/ data/vietnam_celeb/

conda deactivate