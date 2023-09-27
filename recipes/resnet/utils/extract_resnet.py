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


class SpeakerExtractingSegmentSet(Dataset, SegmentSet):
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

        return feature, segment.segmentid



if __name__ == '__main__':
    device = torch.device("cuda")

    extracting_data = SpeakerExtractingSegmentSet(
                                    feature_extractor=Fbank(),
                                    feature_transforms=Linear( [
                                        CMVN(),
                                        Crop(200),
                                    ] ),
                                )


    extracting_data.from_dict(Path("data/voxceleb1/"))

    resnet_model = ResNet()
    resnet_model.load_state_dict(torch.load(sys.argv[1], map_location={"cuda" : "cpu"}).state_dict())
    resnet_model.to(device)
    resnet_model.eval()

    for feat, key in extracting_data:
        feat = feat.unsqueeze(0).unsqueeze(1)
        feat = feat.to(device)

        pred = resnet_model(feat)

        print(key+" "+" ".join(map(str, pred.cpu().detach().numpy()[0])))




