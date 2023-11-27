#!/usr/bin/env python3
import time
from pathlib import Path

import sys

import torch
from torch.utils.data import DataLoader

from kiwano.augmentation import Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop
from kiwano.dataset import SegmentSet
from kiwano.features import Fbank
from kiwano.model import ECAPAModel, ECAPATrainDataset
from recipes.resnet.utils.train_resnet import SpeakerTrainingSegmentSet
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

if __name__ == '__main__':
    training_data = ECAPATrainDataset(
        train_list='data/voxceleb1/liste',
        train_path='db/voxceleb1',
        musan_path='db/musan/musan',
        rir_path='db/rirs_noises',
        num_frames=200
    )

    train_dataloader = DataLoader(training_data, batch_size=400, drop_last=True, shuffle=True, num_workers=10)

    ecapa_tdnn_model = ECAPAModel(
        lr=0.001,
        lr_decay=0.97,
        channel_in=81,
        channel_size=1024,
        n_class=6000,
        loss_margin=0.2,
        loss_scale=30,
        test_step=1
    )

    num_iterations = 10
    print(f"START ECAPA-TDNN {num_iterations} iterations")
    sys.stdout.flush()
    save_path = "exps/ecapa"
    eval_list = "db/voxceleb1/voxceleb1_test_v2.txt"
    eval_path = "db/voxceleb1/wav"
    score_file = open(f"{save_path}/score.txt", "a+")
    best_eer = torch.inf
    for epoch in range(1, num_iterations + 1):
        print(f"\t [{epoch} / {num_iterations}]")
        sys.stdout.flush()
        loss, lr, acc = ecapa_tdnn_model.train_network(epoch=epoch, loader=train_dataloader)
        eer, _ = ecapa_tdnn_model.eval_network(eval_list=eval_list, eval_path=eval_path,
                                               feature_extractor=None)
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
