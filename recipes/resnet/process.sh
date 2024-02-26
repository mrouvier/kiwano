#!/bin/bash
#SBATCH --job-name=cn_celeb
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
# #SBATCH --cpus-per-task=8
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

#rm db/cn_celeb/cn-celeb2.zip
#python3 local/extract_cn_celeb.py db/cn_celeb/
#rm -r data/cn_celeb/
#python3 local/prepare_cn_celeb.py db/cn_celeb/ data/cn_celeb/
# mv  data/cn_celeb/CN-Celeb2_flac/wav db/cn_celeb/CN-Celeb2_flac/
#mv data/cn_celeb/CN-Celeb_flac/dev/wav db/cn_celeb/CN-Celeb_flac/dev/
#mv data/cn_celeb/CN-Celeb_flac/eval/wav db/cn_celeb/CN-Celeb_flac/eval/

python3 local/prepare_cn_celeb.py --in_data db/cn_celeb/CN-Celeb_flac/eval/lists/ --old_file trials.lst

# python3 local/prepare_cn_celeb.py --in_data data/cn_celeb/ --old_file liste --out_data db/cn_celeb/CN-Celeb2_flac/

# python3 local/extract_vietnam_celeb.py db/vietnam_celeb/
# rm  db/vietnam_celeb/vietnam-celeb.zip
# rm  db/vietnam_celeb/vietnam_celeb.zip
# cat  db/vietnam_celeb/vietnam-celeb-part* >  db/vietnam_celeb/vietnam-celeb.zip
# unzip db/vietnam_celeb/vietnam-celeb.zip -d db/vietnam_celeb/

# rm db/vietnam_celeb/full-dataset.zip
# rm db/vietnam_celeb/zisqymuN
# rm  db/vietnam_celeb/vietnam-celeb.zip
# python3 local/extract_vietnam_celeb.py db/vietnam_celeb/
# zip -F db/vietnam_celeb/vietnam-celeb-part.zip --out db/vietnam_celeb/full-dataset.zip

#rm  db/vietnam_celeb/vietnam-celeb.zip
#zip -F db/vietnam_celeb/vietnam-celeb-part.zip --out db/vietnam_celeb/full-dataset.zip
#unzip db/vietnam_celeb/full-dataset.zip
#mv data/id0* db/vietnam_celeb/data/

#rm -r data/vietnam_celeb/
#python3 local/prepare_vietnam_celeb.py db/vietnam_celeb/ data/vietnam_celeb/

python3 local/prepare_vietnam_celeb.py --in_data db/vietnam_celeb/ --out_data db/vietnam_celeb/ --old_file vietnam-celeb-t.txt

# python3 local/prepare_vietnam_celeb.py --in_data db/vietnam_celeb/ --old_file vietnam-celeb-t.txt

conda deactivate