import torchaudio
import numpy as np
import random

#Check if I do modify the id
class Speed:
    def __new__(self, arr: np.ndarray, sample_rate: int):
        speeds = [
                ["speed", 1.0],
                ["speed", 0.9],
                ["speed", 1.1]
            ]
        s = s[ random.randint(0, 2) ]
        #Check if speed == 1.0 and do nothing
        speech, _ = torchaudio.sox_effects.apply_effects_tensor(arr, sample_rate, effects=[s])
        return arr


class Noise:
    def __new__(self, arr: np.ndarray, sample_rate: int):
        return arr
        

class Codec:
    def __new__(self, arr: np.ndarray, sample_rate: int):
        codec = [
            ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8 bit mu-law"),
            ({"format": "wav", "encoding": 'ALAW', "bits_per_sample": 8}, "8 bit a-law"),
            ({"format": "mp3", "compression": -9}, "MP3"),
            ({"format": "vorbis", "compression": -1}, "Vorbis")
            ]
        param, title = random.choice(codec)
        speech = torchaudio.functional.apply_codec(arr, sample_rate, **param)
        return speech


class Reverb:
    def __new__(self, arr: np.ndarray, sample_rate: int):


class Filtering:
    def __new__(self, arr: np.ndarray, sample_rate: int):
        effects = [
                ["bandpass", "2000", "3500"],
                ["bandstop", "200", "500"]
            ]
        e = effects[random.randint(0, 1)]
        speech, _ = torchaudio.sox_effects.apply_effects_tensor(arr, sample_rate, effects=[e])
        return speech


class SpecAugment:
    def __new__(self, arr: np.ndarray, num_t_mask=1: int, num_f_mask=1: int, max_t=10: int, max_f=8: int, prob=0.6: float):


