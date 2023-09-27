#Prepare Voxceleb1
python3 local/download_voxceleb1.py db/voxceleb1/
python3 local/prepare_voxceleb1.py db/voxceleb1/ data/voxceleb1/

#Prepare Voxceleb2
python3 local/download_voxceleb2.py db/voxceleb2/
python3 local/prepare_voxceleb2.py db/voxceleb2/ data/voxceleb2/

#Prepare MUSAN
python3 local/download_musan.py db/musan/
python3 local/prepare_musan.py db/musan/ data/musan/

#Prepare RIRS NOISES
#python3 local/download_rirs_noises.py db/rirs_noises/
#python3 local/prepare_rirs_noises.py db/musan/ data/rirs_noises


#Train resnet
python3 utils/train_resnet.py data/voxceleb1/
python3 utils/extract_resnet.py exp/model3500.mat > exp/xvector.txt
