#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
from torch import nn

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop
from kiwano.dataset import Segment, SegmentSet

import soundfile as sf

from torch.utils.data import DataLoader, Sampler


class SpeakerTrainingSegmentSet():
    def __init__(self, audio_transforms: List[Augmentation] = None, feature_transforms: List[Augmentation] = None):
        self.segments = {}
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.acoustic_strategy = Fbank()

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        segment = None
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)

        audio, sample_rate = segment.load_audio()
        if self.audio_transforms != None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)

        feature = self.acoustic_strategy.extract(audio, sampling_rate=sample_rate)

        if self.feature_transforms != None:
            feature = self.feature_transforms(feature)
        return feature


    def from_dict(self, target_dir: Pathlike):
        with open(target_dir / "liste") as f:
            for line in f:
                segmentid, spkid, duration, audio = line.strip().split(" ")
                self.segments[segmentid] = Segment(segmentid, spkid, (float)(duration), audio)

    def summarize(self):
        print(len(self.segments))



class Toto():
    def __init__(self):
        print("oki")


    def __call__(self, *args):
        if len(args) == 2:
            return 3, 4
        else:
            return 9


if __name__ == '__main__':

    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")


    #s = SegmentSet()
    s = SpeakerTrainingSegmentSet( audio_transforms=Sometimes( [
                                        Noise(musan_music, snr_range=[5,15]),
                                        Noise(musan_speech, snr_range=[13,20]),
                                        Noise(musan_noise, snr_range=[0,15]),
                                        Codec(),
                                        Filtering(),
                                        Normal(),
                                    ] ),
                                feature_transforms=Linear( [
                                        CMVN(),
                                        Crop(300),
                                    ] ),
                                )


    s.from_dict(Path("data/voxceleb1/"))

    print(s[0])


    #n = Noise( musan, snr_range=[10,15] )
    #n = Filtering()
    #n = Codec()

    #arr, sample_rate = s[10].load_audio()


    #arr, sr = n(arr, sample_rate)

    #print( arr )




