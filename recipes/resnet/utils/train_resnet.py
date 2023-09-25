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
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNet

import soundfile as sf

from torch.utils.data import Dataset, DataLoader, Sampler


class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    def __init__(self, audio_transforms: List[Augmentation] = None, feature_extractor = None, feature_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        segment = None
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

        return feature, self.labels[ segment.spkid ]



if __name__ == '__main__':

    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")


    training_data = SpeakerTrainingSegmentSet(
                                    audio_transforms=Sometimes( [
                                        Noise(musan_music, snr_range=[5,15]),
                                        Noise(musan_speech, snr_range=[13,20]),
                                        Noise(musan_noise, snr_range=[0,15]),
                                        Codec(),
                                        Filtering(),
                                        Normal(),
                                    ] ),
                                    feature_extractor=Fbank(),
                                    feature_transforms=Linear( [
                                        CMVN(),
                                        Crop(300),
                                    ] ),
                                )


    training_data.from_dict(Path("data/voxceleb1/"))

    train_dataloader = DataLoader(training_data, batch_size=128, drop_last=True, shuffle=True, num_workers=10)
    iterator = iter(train_dataloader)

    resnet_model = ResNet()

    optimizer = torch.optim.SGD(resnet_model.parameters(), lr=0.2, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    num_iterations = 150000
    for iterations in range(0, num_iterations):
        feats, iden = next(iterator)
        feats = feats.unsqueeze(1)

        preds = resnet_model(feats, iden)

        loss = criterion(preds, iden)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        msg = "{}: [{}/{}] \t C-Loss:{:.4f}".format(time.ctime(), iterations, num_iterations, loss.item())
        print(msg)








     
