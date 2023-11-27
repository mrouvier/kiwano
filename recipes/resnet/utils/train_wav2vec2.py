import argparse
import os
import pdb
import time
from pathlib import Path

import hostlist
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import Wav2Vec2Tokenizer

from kiwano.dataset import SegmentSet
from kiwano.model import ECAPAModel
from kiwano.model.wav2vec2 import CustomWav2Vec2Model
from recipes.resnet.utils.train_resnet import SpeakerTrainingSegmentSet
import pdb
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop, \
    SpecAugment, Reverb, CropWaveForm
import torch
import sys
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    print("START Loading data")
    sys.stdout.flush()
    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    model_name = "facebook/wav2vec2-base-960h"
    model_wav2vec2 = CustomWav2Vec2Model(model_name, 3)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=Sometimes([
            Noise(musan_music, snr_range=[5, 15]),
            Noise(musan_speech, snr_range=[13, 20]),
            Noise(musan_noise, snr_range=[0, 15]),
            Codec(),
            Filtering(),
            Normal()
        ]),
        feature_extractor=model_wav2vec2
    )
    training_data.from_dict(Path("data/voxceleb1/"))

    # batch_size=400
    train_dataloader = DataLoader(training_data, batch_size=400, drop_last=False, shuffle=False, num_workers=30)
    print("END Loading data")
    sys.stdout.flush()

    wav2vec2_outputs = []

    num_iterations = 10

    ecapa_tdnn_model = ECAPAModel(
        lr=0.001,
        lr_decay=0.97,
        channel_in=768,
        channel_size=1024,
        n_class=6000,
        loss_margin=0.2,
        loss_scale=30,
        test_step=1
    )
    print(f"START ECAPA-TDNN {num_iterations} iterations")
    sys.stdout.flush()
    save_path = "exps"
    eval_list = "db/voxceleb1/voxceleb1_test_v2.txt"
    eval_path = "db/voxceleb1/wav"
    score_file = open(f"{save_path}/score.txt", "a+")
    best_eer = torch.inf
    for epoch in range(1, num_iterations + 1):
        print(f"\t [{epoch} / {num_iterations}]")
        sys.stdout.flush()
        loss, lr, acc = ecapa_tdnn_model.train_network(epoch=epoch, loader=train_dataloader)
        eer, _ = ecapa_tdnn_model.eval_network(eval_list=eval_list, eval_path=eval_path,
                                               feature_extractor=model_wav2vec2)
        if eer < best_eer:
            best_eer = eer
            ecapa_tdnn_model.save_parameters(f"{save_path}/best.model")

        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              "%d epoch, ACC %2.2f%%, EER %2.2f%%, BestEER %2.2f%%" % (epoch, acc, eer, best_eer))
        sys.stdout.flush()

        score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, BestEER %2.2f%%\n" % (
            epoch, lr, loss, acc, eer, best_eer))
        score_file.flush()

    print(f"END ECAPA-TDNN")
    sys.stdout.flush()
