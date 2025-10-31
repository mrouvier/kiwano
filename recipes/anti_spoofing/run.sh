#!/bin/bash

stage=0


#Prepare Voxceleb1
if [ $stage -le 0 ]
then
  python local/download_voxceleb1.py db/voxceleb1/
  python local/prepare_voxceleb1.py db/voxceleb1/ data/voxceleb1/
fi

#Prepare Voxceleb2
if [ $stage -le 1 ]
then
  python local/download_voxceleb2.py db/voxceleb2/
  python local/prepare_voxceleb2.py db/voxceleb2/ data/voxceleb2/
fi

#Prepare MUSAN
if [ $stage -le 2 ]
then
  python local/download_musan.py db/musan/
  python local/prepare_musan.py db/musan/ data/musan/
fi

#Prepare RIRS NOISES
if [ $stage -le 3 ]
then
  python local/download_rirs_noises.py db/rirs_noises/
  python local/prepare_rirs_noises.py db/rirs_noises/ data/rirs_noises
fi


if [ $stage -le 4 ]
then
  python local/prepare_sre24_test.py /lustre/fsn1/projects/rech/hho/uiu35yn/sre20/releases/LDC2020E28/ data/sre24_test/
  python local/prepare_sre24_enrollment.py /lustre/fsn1/projects/rech/hho/uiu35yn/sre20/releases/LDC2020E28/ data/sre24_enrollment/
fi
