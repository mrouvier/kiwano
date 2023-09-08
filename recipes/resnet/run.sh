#Prepare Voxceleb1
python3 local/download_voxceleb1.py db/voxceleb1/
python3 local/prepare_voxceleb1.py db/voxceleb1/ data/voxceleb1/
#python3 utils/utt2spk_to_spk2utt.py data/voxceleb1/utt2spk > data/voxceleb1/spk2utt
#python3 utils/utt2dur.py data/voxceleb1/ > data/voxceleb1/utt2dur


#Prepare Voxceleb2
python3 local/download_voxceleb2.py db/voxceleb2/
python3 local/prepare_voxceleb2.py db/voxceleb2/ data/voxceleb2/
#python3 utils/utt2spk_to_spk2utt.py data/voxceleb2/utt2spk > data/voxceleb2/spk2utt
#python3 utils/utt2dur.py data/voxceleb2/ > data/voxceleb2/utt2dur


#Prepare MUSAN
python3 local/download_musan.py db/musan/
python3 local/prepapre_musan.py db/musan/ data/musan/


#Prepare RIRS NOISES
python3 local/download_rirs_noises.py db/rirs_noises/
python3 local/prepapre_rirs_noises.py db/musan/ data/rirs_noises


python3 utils/train_resnet.py data/voxceleb1/
