import torchaudio
import torch
import numpy as np
import random
import math
from kiwano.dataset import SegmentSet

from typing import List

from scipy import signal

from torch.fft import irfft, rfft



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


class SpeedPerturb(Augmentation):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        sp = torchaudio.transforms.SpeedPerturbation(sample_rate, self.factor)
        return sp(tensor), sample_rate


class Normal(Augmentation):
    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        return tensor, sample_rate


class Noise(Augmentation):
    def __init__(self, segs: SegmentSet, snr_range: List[int]):
        self.segs = segs
        # self.segs.load_audio()
        self.snr_range = snr_range

    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        snr_db = random.randint(self.snr_range[0], self.snr_range[1])
        noise_tensor, noise_sample_rate = self.segs.get_random().load_audio()

        if len(noise_tensor) > len(tensor):
            start = random.randint(0, len(noise_tensor) - len(tensor))
            noise_tensor = noise_tensor[start:start + len(tensor)]
        else:
            n = math.ceil(len(tensor) / len(noise_tensor))
            noise_tensor = noise_tensor.repeat(n)
            start = random.randint(0, len(noise_tensor) - len(tensor))
            noise_tensor = noise_tensor[start:start + len(tensor)]

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
            # ({"format": "mp3", "compression": -9}, "MP3"),
            # ({"format": "vorbis", "compression": -1}, "Vorbis")
        ]

        # {"format": "mp3", "codec_config": CodecConfig(compression_level=9)},
        # {"format": "ogg", "encoder": "vorbis"},
        #

    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        param, title = random.choice(self.codec)
        speech = torchaudio.functional.apply_codec(tensor.unsqueeze(0), sample_rate, **param)

        return speech[0], sample_rate


_NEXT_FAST_LEN = {}

def next_fast_len(size):
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1



def convolve1d(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    assert (
            signal.ndim == 1 and kernel.ndim == 1
            ), "signal and kernel must be 1-dimensional"
    m = signal.size(-1)
    n = kernel.size(-1)

    # Compute convolution using fft.
    padded_size = m + n - 1
    # Round up for cheaper fft.
    fast_ftt_size = next_fast_len(padded_size)
    f_signal = rfft(signal, n=fast_ftt_size)
    f_kernel = rfft(kernel, n=fast_ftt_size)
    f_result = f_signal * f_kernel
    result = irfft(f_result, n=fast_ftt_size)

    return result[:padded_size]




class Reverb(Augmentation):
    def __init__(self, segs: SegmentSet):
        self.segs = segs
        self.rir_scaling_factor = 0.5**15

    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        reverb_tensor, reverb_sample_rate = self.segs.get_random().load_audio()
        size = len(tensor)

        power_before = torch.dot(tensor, tensor) / len(tensor)

        #tensor = convolve1d(tensor, reverb_tensor)
        #tensor = torch.Tensor(signal.convolve(tensor, reverb_tensor, mode='full')[:len(tensor)])

        tensor = torch.Tensor(signal.convolve(tensor, reverb_tensor, mode='full')[:size])
        power_after = torch.dot(tensor, tensor) / size
        tensor *= (power_before / power_after).sqrt()
        tensor = torch.clamp(tensor, -1.0, 1.0)

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
        result = tensor[start_time:start_time + self.duration, :]
        return result


class CropNotRandom(Augmentation):
    def __init__(self, duration: int):
        self.duration = duration

    def __call__(self, tensor: torch.Tensor):
        max_start_time = self.duration

        if tensor.shape[0] < self.duration:
            max_start_time = tensor.shape[0]
     
        result = tensor[0:max_start_time, :]
        return result




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
    def __init__(self, norm_means=True, norm_vars=False):
        self.norm_means = norm_means
        self.norm_vars = norm_vars

    def __call__(self, tensor: torch.Tensor):
        if self.norm_means == True:
            tensor = tensor - tensor.mean(dim=0)
        return tensor
