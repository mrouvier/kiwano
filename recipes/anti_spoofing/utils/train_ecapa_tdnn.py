#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from kiwano.augmentation import (
    CMVN,
    Codec,
    Compose,
    Crop,
    Filtering,
    Noise,
    Normal,
    OneOf,
)
from kiwano.dataset import SegmentSet
from kiwano.features import Fbank
from kiwano.model import ECAPAModel
from recipes.resnet.utils.train_resnet import SpeakerTrainingSegmentSet

if __name__ == "__main__":
    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=OneOf(
            [
                Noise(musan_music, snr_range=[5, 15]),
                Noise(musan_speech, snr_range=[13, 20]),
                Noise(musan_noise, snr_range=[0, 15]),
                Codec(),
                Filtering(),
                Normal(),
            ]
        ),
        feature_extractor=Fbank(),
        feature_transforms=Compose([CMVN(), Crop(300)]),
    )

    training_data.from_dict(Path("data/voxceleb1/"))

    train_dataloader = DataLoader(
        training_data, batch_size=128, drop_last=True, shuffle=True, num_workers=10
    )

    ecapa_tdnn_model = ECAPAModel(
        lr=0.001,
        lr_decay=0.97,
        channel_size=1024,
        n_class=6000,
        loss_margin=0.2,
        loss_scale=30,
        test_step=1,
    )

    # num_iterations = 150000
    num_iterations = 2
    loss, lr, acc = ecapa_tdnn_model.train_network(
        epoch=num_iterations, loader=train_dataloader
    )
