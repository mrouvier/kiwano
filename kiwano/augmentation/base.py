import torchaudio
import torch
import numpy as np
import random
import math
from kiwano.dataset import SegmentSet

from typing import List

from scipy import signal



class Augmentation:
    pass


class Pipeline:
    def __init__(self, transforms: List[Augmentation]):
        self.transforms = transforms

class Sometimes(Pipeline):
    def __call__(self, *args):
        transform = random.choice(self.transforms) 
        if len(args) == 1:
            tensor = args[0]
            return transform(tensor)
        else:
            tensor = args[0]
            sample_rate = args[1]
            return transform(tensor, sample_rate)

class Linear(Pipeline):
    def __call__(self, *args):
        if len(args) == 1:
            tensor = args[0]
            for transform in self.transforms:
                tensor = transform(tensor)
            return tensor
        else:
            tensor = args[0]
            sample_rate = args[1]
            for transform in self.transforms:
                tensor, sample_rate = transform(tensor, sample_rate)
            return tensor, sample_rate


'''
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
'''


class Normal(Augmentation):
    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        return tensor, sample_rate



class Noise(Augmentation):
    def __init__(self, segs: SegmentSet, snr_range: List[int]):
        self.segs = segs
        #self.segs.load_audio()
        self.snr_range = snr_range

    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        snr_db = random.randint( self.snr_range[0], self.snr_range[1] )
        noise_tensor, noise_sample_rate = self.segs.get_random().load_audio()

        if len(noise_tensor) > len(tensor):
            start=random.randint(0, len(noise_tensor)-len(tensor))
            noise_tensor=noise_tensor[start:start+len(tensor)]
        else:
            n=math.ceil(len(tensor)/len(noise_tensor))
            noise_tensor=noise_tensor.repeat(n)
            start=random.randint(0, len(noise_tensor)-len(tensor))
            noise_tensor=noise_tensor[start:start+len(tensor)]

        speech_power = torch.linalg.vector_norm(tensor, ord=2) ** 2
        noise_power = torch.linalg.vector_norm(noise_tensor, ord=2) ** 2

        original_snr_db = 10 * (torch.log10(speech_power) - torch.log10(noise_power))
        scale = 10 ** ((original_snr_db - snr_db) / 20.0)
        tensor = scale * noise_tensor + tensor
        
        return tensor, sample_rate
        

class Codec(Augmentation):
    def __init__(self):
        self.codec = [
                ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8 bit mu-law"),
                ({"format": "wav", "encoding": 'ALAW', "bits_per_sample": 8}, "8 bit a-law"),
                #({"format": "mp3", "compression": -9}, "MP3"),
                #({"format": "vorbis", "compression": -1}, "Vorbis")
            ]

        #{"format": "mp3", "codec_config": CodecConfig(compression_level=9)},
        #{"format": "ogg", "encoder": "vorbis"},
        #


    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        param, title = random.choice(self.codec)
        speech = torchaudio.functional.apply_codec(tensor.unsqueeze(0), sample_rate, **param)

        return speech[0], sample_rate


class Reverb(Augmentation):
    def __init__(self, segs: SegmentSet):
        self.segs = segs
        self.rir_scaling_factor = 0.5**15

    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        reverb_tensor, reverb_sample_rate = self.segs.get_random().load_audio()

        power_before_reverb = sum(abs(tensor ** 2)) / len(tensor)

        reverb_tensor *= self.rir_scaling_factor

        tensor = torch.Tensor(signal.convolve(tensor, reverb_tensor, mode='full')[:len(tensor)])

        power_after_reverb = sum(abs(tensor ** 2)) / len(tensor)

        if power_after_reverb > 0:
            tensor *= torch.sqrt(power_before_reverb / power_after_reverb)

        return tensor, sample_rate


class Filtering(Augmentation):
    def __init__(self):
        pass

    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        effects = [
                ["bandpass", "2000", "3500"],
                ["bandreject", "200", "500"]
            ]
        e = effects[random.randint(0, 1)]

        speech, _ = torchaudio.sox_effects.apply_effects_tensor(tensor.unsqueeze(0), sample_rate, effects=[e])

        return speech[0], sample_rate


class Crop(Augmentation):
    def __init__(self, duration: int):
        self.duration = duration

    def __call__(self, tensor: torch.Tensor):
        max_start_time = tensor.shape[0] - self.duration 
        start_time = random.randint(0, max_start_time)
        return tensor[start_time:start_time+self.duration, :]
        return tensor

class SpecAugment(Augmentation):
    def __init__(self,  num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.6):
        self.num_t_mask = num_t_mask
        self.num_f_mask = num_f_mask
        self.max_t = max_t
        self.max_f = max_f
        self.prob = prob

    def __call__(self, tensor: torch.Tensor):
        if random.random() < self.prob:
                max_frames, max_freq = tensor.shape

                # time mask
                for i in range(self.num_t_mask):
                    start = random.randint(0, max_frames - 1)
                    length = random.randint(1, self.max_t)
                    end = min(max_frames, start + length)
                    tensor[start:end, :] = 0

                # freq mask
                for i in range(self.num_f_mask):
                    start = random.randint(0, max_freq - 1)
                    length = random.randint(1, self.max_f)
                    end = min(max_freq, start + length)
                    tensor[:, start:end] = 0

        return tensor


class CMVN(Augmentation):
    def __init__(self, norm_means = True, norm_vars = False):
        self.norm_means = norm_means
        self.norm_vars = norm_vars

    def __call__(self, tensor: torch.Tensor):
        if self.norm_means == True:
            tensor = tensor - tensor.mean(dim=0)
        return tensor

