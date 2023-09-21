import torchaudio
import numpy as np
import random
from kiwano.dataset import SegmentSet

from typing import List



class Augmentation:
    pass



class Pipeline:
    def __init__(self, transforms: List[Augmentation]):
        self.transforms = transforms

class Sometimes(Pipeline):
    def __call__(self, arr: np.ndarray, sample_rate: int):
        transform = random.choice(self.transforms) 
        return transform(arr, sample_rate)

    def __call__(self, arr: np.ndarray):
        transform = random.choice(self.transforms) 
        return transform(arr)


class Linear(Pipeline):
    def __call__(self, arr: np.ndarray, sample_rate: int):
        for transform in self.transforms:
            arr, sample_rate = transform(arr, sample_rate)
        return arr, sample_rate

    def __call__(self, arr: np.ndarray):
        for transform in self.transforms:
            arr = transform(arr)
        return arr




#Check if I do modify the id
class Speed(Augmentation):
    def __init__(self):
        self.speeds = [
                ["speed", 1.0],
                ["speed", 0.9],
                ["speed", 1.1]
            ]

    def __call__(self, arr: np.ndarray, sample_rate: int):
        speed = self.speeds[ random.randint(0, 2) ]
        #Check if speed == 1.0 and do nothing
        speech, _ = torchaudio.sox_effects.apply_effects_tensor(arr, sample_rate, effects=[s])
        return arr


class Noise(Augmentation):
    def __init__(self, segs: SegmentSet, snr_range: List[int]):
        self.segs = segs
        self.segs.load_audio()
        self.snr_range = snr_range

    def __call__(self, arr: np.ndarray, sample_rate: int):
        return arr
        

class Codec(Augmentation):
    def __init__(self):
        self.codec = [
            ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8 bit mu-law"),
            ({"format": "wav", "encoding": 'ALAW', "bits_per_sample": 8}, "8 bit a-law"),
            ({"format": "mp3", "compression": -9}, "MP3"),
            ({"format": "vorbis", "compression": -1}, "Vorbis")
            ]


    def __call__(self, arr: np.ndarray, sample_rate: int):
        param, title = random.choice(self.codec)
        speech = torchaudio.functional.apply_codec(arr, sample_rate, **param)
        return speech


class Reverb(Augmentation):
    def __init__(self):
        pass

    def __call__(self, arr: np.ndarray, sample_rate: int):
        pass


class Filtering(Augmentation):
    def __init__(self):
        pass

    def __call__(self, arr: np.ndarray, sample_rate: int):
        effects = [
                ["bandpass", "2000", "3500"],
                ["bandstop", "200", "500"]
            ]
        e = effects[random.randint(0, 1)]
        speech, _ = torchaudio.sox_effects.apply_effects_tensor(arr, sample_rate, effects=[e])
        return speech

'''
class SpecAugment(Augmentation):
    def __init__(self, num_t_mask=1: int, num_f_mask=1: int, max_t=10: int, max_f=8: int, prob=0.6: float):
        pass

    def __call__(self, arr: np.ndarray):
        pass
'''


class Crop(Augmentation):
    def __init__(self, duration: int):
        self.duration = duration

    def __call__(self, arr: np.ndarray):
        pass




