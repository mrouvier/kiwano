#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional
from kiwano.utils import Pathlike
from kiwano.features import Fbank
from typing import Union
import librosa

from torch.utils.data import DataLoader, Sampler


#add_reverb,add_noise,filtering,phone_filtering,codec

#add_reverb,add_noise,phone_filtering,codec



'''
train_dataset = K2SpeechRecognitionDataset(
    cut_transforms=[
        PerturbSpeed(factors=[0.9, 1.1], p=2 / 3),
        PerturbVolume(scale_low=0.125, scale_high=2.0, p=0.5),
        # [optional] you can supply noise examples to be mixed in the data
        CutMix(musan_cuts, snr=[10, 20], p=0.5),
        # [optional] you can supply RIR examples to reverberate the data
        ReverbWithImpulseResponse(rir_recordings, p=0.5),
    ],
    input_transforms=[
        SpecAugment(),  # default configuration is well-tuned
    ],
    input_strategy=OnTheFlyFeatures(Fbank()),
)
'''

class Augmentation():

    def __init__(self):







class Segment():
    segmentid: str
    duration: float
    spkid: str
    file_path: str

    def __init__(self, segmentid : str, spkid : str, duration : float, file_path : str):
        self.segmentid = segmentid
        self.spkid = spkid
        self.duration = duration
        self.file_path = file_path

    def compute_features(self):
        audio_data, sample_rate = librosa.load(self.file_path)
        fb = Fbank()
        return fb.extract(audio_data, sampling_rate=16000)



class SegmentSet():
    def __init__(self):
        self.segments = {}

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        print("ttt")
        if isinstance(segment_id_or_index, str):
            return self.segments[segment_id_or_index]
        return next( val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index )

    def from_dict(self, target_dir: Pathlike):
        with open(target_dir / "liste") as f:
            for line in f:
                segmentid, spkid, duration, audio = line.strip().split(" ")
                self.segments[segmentid] = Segment(segmentid, spkid, (float)(duration), audio)


    def summarize(self):
        print(len(self.segments))

if __name__ == '__main__':
    s = SegmentSet()
    s.from_dict(Path("data/voxceleb1/"))
    f = s["id10001_J9lHsKG98U8_00007"].compute_features()
    print(f)
    print(f.shape)



