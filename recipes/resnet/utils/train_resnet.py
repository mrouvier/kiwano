#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
import time
from torch import nn

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop, \
    SpecAugment, Reverb
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNet, SpkScheduler

import soundfile as sf

from torch.utils.data import Dataset, DataLoader, Sampler


class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    def __init__(self, audio_transforms: List[Augmentation] = None, feature_extractor=None,
                 feature_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]):
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)

        audio, sample_rate = segment.load_audio()
        if self.audio_transforms != None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)

        if self.feature_extractor != None:
            feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)

        if self.feature_transforms != None:
            feature = self.feature_transforms(feature)

        return feature, self.labels[segment.spkid]


if __name__ == '__main__':
    device = torch.device("cuda")

    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    reverb = SegmentSet()
    reverb.from_dict(Path("data/rirs_noises/"))

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=Sometimes([
            Noise(musan_music, snr_range=[5, 15]),
            Noise(musan_speech, snr_range=[13, 20]),
            Noise(musan_noise, snr_range=[0, 15]),
            Codec(),
            Filtering(),
            Normal(),
            Reverb(reverb)
        ]),
        feature_extractor=Fbank(),
        feature_transforms=Linear([
            CMVN(),
            Crop(350),
            SpecAugment(),
        ]),
    )

    training_data.from_dict(Path("data/voxceleb2/"))

    train_dataloader = DataLoader(training_data, batch_size=48, drop_last=True, shuffle=True, num_workers=10)
    iterator = iter(train_dataloader)

    resnet_model = ResNet()

    resnet_model.to(device)

    optimizer = torch.optim.SGD(resnet_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    spk_scheduler = SpkScheduler(optimizer, num_epochs=150, initial_lr=0.1, final_lr=0.00005, warm_up_epoch=6)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epochs in range(0, 150):
        iterations = 0
        for feats, iden in train_dataloader:

            feats = feats.unsqueeze(1)

            feats = feats.float().to(device)
            iden = iden.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                preds = resnet_model(feats, iden)
                loss = criterion(preds, iden)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step()

            if iterations % 100 == 0:
                msg = "{}: [{}/{}] {} \t C-Loss:{:.4f} \t LR : {:.8f}".format(time.ctime(), epochs, 150, iterations,
                                                                              loss.item(), spk_scheduler.get_current_lr())
                print(msg)

            iterations += 1

        spk_scheduler.step()
        torch.save(resnet_model.state_dict(), "exp/resnet_noddp/model" + str(epochs) + ".mat")
