#!/bin/bash

stage=0


#Prepare Voxceleb1
if [ $stage -le 0 ]
then
    python -m kiwano.corpus.cli download voxceleb1 db/voxceleb1/
    python -m kiwano.corpus.cli prepare voxceleb1 db/voxceleb1/ data/voxceleb1/
fi

#Prepare Voxceleb2
if [ $stage -le 1 ]
then
    python -m kiwano.corpus.cli download voxceleb2 db/voxceleb2/
    python -m kiwano.corpus.cli prepare voxceleb2 db/voxceleb2/ data/voxceleb2/

fi

#Prepare MUSAN
if [ $stage -le 2 ]
then
    python -m kiwano.corpus.cli download musan db/musan/
    python -m kiwano.corpus.cli prepare musan db/musan/ data/musan/
fi

#Prepare RIRS NOISES
if [ $stage -le 3 ]
then
    python -m kiwano.corpus.cli download rirs_noises db/rirs_noises/
    python -m kiwano.corpus.cli prepare rirs_noises db/rirs_noises/ data/rirs_noises/
fi
