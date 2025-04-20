import torchaudio
import torch
import torch.nn as nn
import numpy as np
import random
import math
import copy

from kiwano.dataset import SegmentSet

from typing import List, Tuple

from scipy import signal

import torch.nn.functional as F


class Augmentation:
    pass


class Pipeline:
    def __init__(self, transforms: List[Augmentation]) -> None:
        self.transforms = transforms


class SomeOf(Pipeline):
    """
    This class implements a randomized transformation pipeline that applies
    a subset of the provided augmentations.

    At each call, a random number of transformations (specified by an integer
    or a range) is applied to the input data. The transformations are chosen
    randomly from the list without replacement.

    This augmentation is inspired by the SomeOf transform from the
    Audiomentation library (https://github.com/iver56/audiomentations).

    Arguments
    ---------
    num_transforms : int or tuple
        If an integer, exactly that many transforms are applied.
        If a tuple of (min, max), a random number between min and max (inclusive)
        is used to select how many transforms to apply.
    transforms : list of Augmentation
        A list of callable augmentation objects to be sampled and applied.

    Example
    -------
    >>> transform1 = Normal()
    >>> transform2 = Normal()
    >>> aug = SomeOf(num_transforms=(1, 2), transforms=[transform1, transform2])
    >>> x = torch.randn(40, 100)
    >>> x_aug = aug(x)

    >>> # If transforms require sample rate as well
    >>> x = torch.randn(1, 16000)
    >>> sr = 16000
    >>> x_aug, sr_aug = aug(x, sr)
    """

    def __init__(
        self, num_transforms: int or Tuple, transforms: List[Augmentation]
    ) -> None:
        self.num_transforms = num_transforms
        self.transforms = transforms

    def __call__(self, *args):
        if type(self.num_transforms) == tuple:
            if len(self.num_transforms) == 1:
                num_transforms_to_apply = random.randint(
                    self.num_transforms[0], len(self.transforms)
                )
            else:
                num_transforms_to_apply = random.randint(
                    self.num_transforms[0], self.num_transforms[1]
                )
        else:
            num_transforms_to_apply = self.num_transforms

        all_transforms_indexes = list(range(len(self.transforms)))
        transform_indexes = sorted(
            random.sample(all_transforms_indexes, num_transforms_to_apply)
        )

        if len(args) == 1:
            tensor = args[0]
            for transform_index in transform_indexes:
                tensor = self.transforms[transform_index](tensor)
            return tensor
        else:
            tensor = args[0]
            sample_rate = args[1]
            for transform_index in transform_indexes:
                tensor, sample_rate = self.transforms[transform_index](
                    tensor, sample_rate
                )
            return tensor, sample_rate


class OneOf(Pipeline):
    """
    This class implements a pipeline that randomly selects one transform from a list
    and applies it to the input.

    This is useful in data augmentation scenarios where you want to apply one of
    several possible transformations to the input data with equal probability.

    Arguments
    ---------
    transforms : list
        A list of Augementation transforms. Each transform should accept either one or two arguments,
        depending on the context (e.g., with or without sample rate).

    Example
    -------
    >>> transform1 = SpeedPerturb()
    >>> transform2 = Normal()
    >>> pipeline = OneOf(transforms=[transform1, transform2])
    >>> result = pipeline(torch.tensor([1.0]))
    """

    def __call__(self, *args):
        transform = random.choice(self.transforms)
        if len(args) == 1:
            tensor = args[0]
            return transform(tensor)
        else:
            tensor = args[0]
            sample_rate = args[1]
            return transform(tensor, sample_rate)


class Compose(Pipeline):
    """
    This class implements a transformation composition module for
    audio or tensor data preprocessing.

    It applies a sequence of transformations defined in the pipeline to the input.
    This is typically used to build complex preprocessing steps by combining
    multiple transform operations.

    Arguments
    ---------
    transforms : list
        A list of transformation functions or modules to apply sequentially.
        Each transform should support either one or two arguments depending on the mode.

    Usage
    -----
    If a single tensor is passed, each transform should accept and return a tensor.
    If both tensor and sample rate are passed, each transform should accept and return
    a tuple (tensor, sample_rate).

    Example
    -------
    >>> transforms = [Normalize(), Resample(16000)]
    >>> pipeline = Compose(transforms)
    >>> waveform = torch.rand([1, 16000])
    >>> output = pipeline(waveform)
    >>> waveform, sample_rate = torch.rand([1, 16000]), 16000
    >>> output_waveform, output_sr = pipeline(waveform, sample_rate)
    """

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


def next_power_of_2(x: int) -> int:
    """
    Return the smallest power of 2 greater than or equal to x.
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class AirAbsorption(Augmentation):
    air_absorption_table = {
        "10C_30-50%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0, 91.5, 289.0]
        ],
        "10C_50-70%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1, 76.6, 280.2]
        ],
        "10C_70-90%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8, 58.0, 214.9]
        ],
        "20C_30-50%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3, 72.3, 259.9]
        ],
        "20C_50-70%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5, 44.4, 148.7]
        ],
        "20C_70-90%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6, 31.2, 93.8]
        ],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
    }

    def __init__(
        self,
        min_temperature: float = 10.0,
        max_temperature: float = 20.0,
        min_humidity: float = 30.0,
        max_humidity: float = 90.0,
        min_distance: float = 10.0,
        max_distance: float = 100.0,
    ):
        """
        if float(min_temperature) not in [10.0, 20.0] or float(max_temperature) not in [10.0, 20.0]:
            raise ValueError("Temperature must be either 10.0 or 20.0")
        if min_temperature > max_temperature:
            raise ValueError("min_temperature must not exceed max_temperature")
        if not (30 <= min_humidity <= max_humidity <= 90):
            raise ValueError("Humidity bounds must be within [30,90] and min_humidity <= max_humidity")
        if not (0 < min_distance <= max_distance <= 2500.0):
            raise ValueError("Distance must be within (0.0, 2500.0]")
        """

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.min_humidity = int(min_humidity)
        self.max_humidity = int(max_humidity)
        self.min_distance = min_distance
        self.max_distance = max_distance

    def _get_temperature_humidity_key(self, temperature: float, humidity: float) -> str:
        key = str(int(temperature)) + "C"
        bounds = [30, 50, 70, 90]
        for n in range(1, len(bounds)):
            if bounds[n - 1] <= humidity <= bounds[n]:
                key += f"_{bounds[n-1]}-{bounds[n]}%"
                break
        return key

    def _interpolate(
        self, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
    ) -> torch.Tensor:
        # Linear interpolation for 1D tensors
        # xp must be sorted
        idx = torch.searchsorted(xp, x)
        idx = idx.clamp(1, xp.size(0) - 1)
        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]
        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (x - x0)

    def __call__(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        temperature = 10 * torch.randint(
            int(self.min_temperature) // 10, int(self.max_temperature) // 10 + 1
        )
        humidity = torch.randint(self.min_humidity, self.max_humidity + 1)
        distance = torch.uniform(self.min_distance, self.max_distance)

        key = self._get_temperature_humidity_key(temperature, humidity)
        coeffs = torch.tensor(self.air_absorption_table[key])
        attenuation_values = torch.exp(-distance * coeffs)

        # FFT setup
        first_bw = self.air_absorption_table["center_freqs"][0] / (2**0.5)
        n_fft = next_power_of_2(int(sample_rate / 2 / first_bw))
        # frequencies for rfft
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        # target attenuations in dB then linear
        db_coeffs = 20 * torch.log10(attenuation_values)
        center_freqs = torch.tensor(self.air_absorption_table["center_freqs"])
        db_target = self._interpolate(freqs, center_freqs, db_coeffs)
        linear_target = 10 ** (db_target / 20)

        # STFT
        stft = torch.stft(
            samples, n_fft=n_fft, return_complex=True
        )  # shape: (channels, freq, time)

        # apply mask
        mask = linear_target.unsqueeze(1).expand(-1, stft.size(-1))
        stft_filtered = stft * mask

        # inverse STFT
        result = torch.istft(stft_filtered, n_fft=n_fft, length=samples.shape[0])


class AirAbsorption(Augmentation):
    """
    This class implements a simplified air absorption augmentation
    for audio signals based on distance, temperature, and humidity.

    The attenuation is computed using a heuristic model where high
    frequencies are more strongly absorbed. The model takes into
    account environmental parameters like air humidity and temperature,
    and simulates signal degradation over distance.

    Arguments
    ---------
    min_distance : float
        Minimum distance for the simulated propagation (in meters).
    max_distance : float
        Maximum distance for the simulated propagation (in meters).
    min_temperature : float
        Minimum air temperature (in degrees Celsius).
    max_temperature : float
        Maximum air temperature (in degrees Celsius).
    min_humidity : float
        Minimum relative humidity (in percentage, range [0, 100]).
    max_humidity : float
        Maximum relative humidity (in percentage, range [0, 100]).
    num_bands : int
        Number of frequency bands to use for attenuation modeling.

    Example
    -------
    >>> audio = torch.randn(1, 48000)
    >>> aug = AirAbsorption(min_distance=10.0, max_distance=100.0)
    >>> augmented_audio, sr = aug(audio, sample_rate=48000)
    >>> augmented_audio.shape
    torch.Size([1, 48000])
    """

    def __init__(
        self,
        min_distance: float = 10.0,
        max_distance: float = 100.0,
        min_temperature: float = 10.0,
        max_temperature: float = 20.0,
        min_humidity: float = 30.0,
        max_humidity: float = 90.0,
        num_bands: int = 32,
    ) -> None:

        assert (
            min_distance <= max_distance
        ), "min_distance must be less than or equal to max_distance"
        assert (
            min_temperature <= max_temperature
        ), "min_temperature must be less than or equal to max_temperature"
        assert (
            min_humidity <= max_humidity
        ), "min_humidity must be less than or equal to max_humidity"
        assert min_distance >= 0, "min_distance must be non-negative."
        assert 0 <= min_humidity <= 100, "min_humidity must be in [0, 100]."
        assert 0 <= max_humidity <= 100, "max_humidity must be in [0, 100]."

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.min_humidity = min_humidity
        self.max_humidity = max_humidity
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_bands = num_bands

    def _compute_absorption_coeffs(
        self, freqs: torch.Tensor, temperature: float, humidity: float
    ) -> torch.Tensor:
        # Simplified heuristic: high freqs absorb more
        temp_factor = 0.003 * (temperature - 20.0)
        hum_factor = 0.002 * (humidity - 50.0)

        # Base absorption ~ f^1.5 scaled for simplicity
        base_absorption = 1e-5 * (freqs**1.5)
        coeffs = base_absorption * (1 + temp_factor + hum_factor)
        return coeffs

    def __call__(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        distance = random.uniform(self.min_distance, self.max_distance)
        humidity = random.uniform(self.min_humidity, self.max_humidity)
        temperature = random.uniform(self.min_temperature, self.max_temperature)

        # STFT
        stft = torch.stft(samples, n_fft=2048, hop_length=512, return_complex=True)
        mag, phase = torch.abs(stft), torch.angle(stft)

        print(mag.shape)
        print(phase.shape)

        freqs = torch.fft.rfftfreq(2048, d=1.0 / sample_rate)
        attenuation = torch.exp(
            -distance * self._compute_absorption_coeffs(freqs, temperature, humidity)
        )

        # Apply attenuation to each frequency bin
        mag_filtered = mag * attenuation[None, :, None]

        # Reconstruct waveform
        stft_filtered = mag_filtered * torch.exp(1j * phase)
        print(stft_filtered)
        print(stft_filtered.shape)
        print(samples.shape[0])
        waveform_filtered = torch.istft(
            stft_filtered, n_fft=2048, hop_length=512, length=samples.shape[0]
        )

        return waveform_filtered, sample_rate


class AddGaussianNoise(Augmentation):
    """
    Adds Gaussian noise to an audio signal with a randomly chosen standard deviation
    between min_amplitude and max_amplitude.

    Arguments
    ---------
    min_amplitude : float
        Minimum standard deviation of the Gaussian noise.
    max_amplitude : float
        Maximum standard deviation of the Gaussian noise.

    Example
    -------
    >>> samples = torch.randn(16000)
    >>> sr = 16000
    >>> augment = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015)
    >>> augmented_samples, sr = augment(samples, sr)
    >>> augmented_samples.shape
    torch.Size([16000])

    """

    def __init__(
        self, min_amplitude: float = 0.001, max_amplitude: float = 0.015
    ) -> None:
        assert (
            0 <= min_amplitude <= max_amplitude
        ), "min_amplitude must be non-negative and less than or equal to max_amplitude."

        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def __call__(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        std = random.uniform(self.min_amplitude, self.max_amplitude)
        noise = torch.randn_like(samples) * std

        noisy_waveform = samples + noise
        return noisy_waveform, sample_rate


class ClippingDistortion(Augmentation):
    """
    This class implements a clipping-based audio distortion augmentation based on
    percentile thresholding.

    The transformation clips the audio signal by computing lower and upper thresholds
    from random percentiles and clamping the signal accordingly. This simulates
    distortion effects by attenuating extreme sample values.

    This augmentation is inspired by the ClippingDistortion transform from the
    Audiomentation library (https://github.com/iver56/audiomentations).

    Arguments
    ---------
    min_percentile_threshold : int
        Minimum percentile range to apply for clipping (default: 0).
    max_percentile_threshold : int
        Maximum percentile range to apply for clipping (default: 40).

    Example
    -------
    >>> samples = torch.randn(16000)
    >>> sr = 16000
    >>> augment = ClippingDistortion(min_percentile_threshold=10, max_percentile_threshold=30)
    >>> augmented_samples, sr = augment(samples, sr)
    >>> augmented_samples.shape
    torch.Size([16000])
    """

    def __init__(
        self, min_percentile_threshold: int = 0, max_percentile_threshold: int = 40
    ) -> None:
        assert (
            min_percentile_threshold <= max_percentile_threshold
        ), "Minimum percentile threshold must be less than or equal to the maximum percentile threshold."
        assert (
            0 <= min_percentile_threshold <= 100
        ), "Minimum percentile threshold must be between 0 and 100."
        assert (
            0 <= max_percentile_threshold <= 100
        ), "Maximum percentile threshold must be between 0 and 100."

        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold

    def torch_percentile(self, input_tensor: torch.Tensor, q: float) -> torch.Tensor:
        k = int((q / 100.0) * (input_tensor.numel() - 1))
        k = max(min(k, input_tensor.numel() - 1), 0)
        sorted_vals, _ = torch.sort(input_tensor)
        return sorted_vals[k]

    def __call__(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        threshold = random.randint(
            self.min_percentile_threshold, self.max_percentile_threshold
        )
        lower_p = threshold / 2
        upper_p = 100 - lower_p

        lower_thresh = self.torch_percentile(samples, lower_p)
        upper_thresh = self.torch_percentile(samples, upper_p)

        return (
            torch.clamp(samples, min=lower_thresh.item(), max=upper_thresh.item()),
            sample_rate,
        )


class Aliasing(Augmentation):
    def __init__(self, original_sr: int, target_sr: int) -> None:
        assert (
            target_sr < original_sr
        ), "Target sample rate must be lower than original sample rate."
        self.original_sr = original_sr
        self.target_sr = target_sr

        self.target_mel = self.convert_frequency_to_mel(self.target_sr)
        self.original_mel = self.convert_frequency_to_mel(self.original_sr)

    def convert_frequency_to_mel(self, f: float) -> float:
        # Convert f hertz to mels : https://en.wikipedia.org/wiki/Mel_scale#Formula
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def convert_mel_to_frequency(self, m: float) -> float:
        # Convert m mels to hertz : https://en.wikipedia.org/wiki/Mel_scale#History_and_other_formulas
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    def __call__(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        orig_len = samples.shape[0]
        new_sample_rate = self.convert_mel_to_frequency(
            random.uniform(self.target_mel, self.original_mel)
        )

        down_len = int(orig_len * new_sample_rate / self.original_sr)

        # Downsample by linear interpolation (no low-pass filtering)
        downsampled = F.interpolate(
            samples.unsqueeze(0).unsqueeze(0),
            size=down_len,
            mode="linear",
            align_corners=True,
        )

        # Upsample back to original length
        upsampled = (
            F.interpolate(downsampled, size=orig_len, mode="linear", align_corners=True)
            .squeeze(0)
            .squeeze(0)
        )

        return upsampled, sample_rate


class SpeedPerturb(Augmentation):
    """
    This class implements speed perturbation as a data augmentation method
    for audio processing tasks.

    Speed perturbation modifies the playback speed of an audio signal,
    thereby altering its temporal properties without affecting the pitch
    when followed by a rate correction. This technique is commonly used
    to improve robustness and generalization in speech models.

    Reference: Speaker Augmentation and Bandwidth Extension for Deep Speaker Embedding,
    Yamamoto et al., https://www.isca-archive.org/interspeech_2019/yamamoto19_interspeech.html

    Arguments
    ---------
    factor : float
        The factor by which the speed is perturbed. A value > 1.0 increases
        speed (shorter duration), while a value < 1.0 decreases speed
        (longer duration).

    Example
    -------
    >>> audio_tensor = torch.rand([16000])  # 1 second of dummy audio at 16kHz
    >>> sample_rate = 16000
    >>> augmenter = SpeedPerturb(factor=1.1)
    >>> augmented_tensor, new_sample_rate = augmenter(audio_tensor, sample_rate)
    >>> augmented_tensor.shape
    torch.Size([...])  # varies depending on speed factor
    """

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(
        self, tensor: torch.Tensor, sample_rate: int
    ) -> Tuple[torch.Tensor, int]:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            tensor.unsqueeze(0),
            sample_rate,
            [["speed", str(self.factor)], ["rate", str(sample_rate)]],
        )
        return wav[0], sample_rate


class Normal(Augmentation):
    """
    This class implements a no-op (identity) audio augmentation that returns
    the input tensor unchanged. It serves as a placeholder or default
    augmentation in audio processing pipelines.

    Arguments
    ---------
    tensor : torch.Tensor
        Input audio tensor to be returned as-is.
    sample_rate : int
        Sampling rate of the input audio (unused in this operation).

    Returns
    -------
    tensor : torch.Tensor
        The same input tensor.
    sample_rate : int
        The same input sample rate.

    Example
    -------
    >>> aug = Normal()
    >>> waveform = torch.rand([1, 16000])
    >>> sr = 16000
    >>> out_tensor, out_sr = aug(waveform, sr)
    >>> torch.equal(waveform, out_tensor)
    True
    >>> sr == out_sr
    True
    """

    def __call__(
        self, tensor: torch.Tensor, sample_rate: int
    ) -> Tuple[torch.Tensor, int]:
        return tensor, sample_rate


class B12Attack(Augmentation):
    """
    This class implements the B12Attack augmentation technique, which
    randomly segments and shuffles parts of an audio signal.

    This method retains only the segments with sufficient energy
    (based on an amplitude threshold), and then randomly permutes them
    to generate a new augmented version of the signal.

    Reference: Data augmentations for audio deepfake detection for the ASVspoof5 closed condition,
    Duroselle et al., https://hal.science/hal-04770832/

    Arguments
    ---------
    mu : float
        Mean duration (in seconds) of each audio segment (default: 8e-2).
    sigma2 : float
        Variance of the segment duration (default: 2.5e-5).
    amplitude_threshold : float
        Minimum average amplitude for a segment to be retained (default: 5e-2).

    Example
    -------
    >>> audio = torch.randn(16000)  # 1 second of audio at 16 kHz
    >>> aug = B12Attack(mu=0.08, sigma2=2.5e-5, amplitude_threshold=0.05)
    >>> augmented_audio, sr = aug(audio, sample_rate=16000)
    >>> augmented_audio.shape
    torch.Size([...])
    """

    def __init__(
        self,
        mu: float = 8e-2,
        sigma2: float = 2.5e-5,
        amplitude_threshold: float = 5e-2,
    ) -> None:
        self.mu = mu
        self.sigma2 = sigma2
        self.amplitude_threshold = amplitude_threshold

    def __call__(
        self, tensor: torch.Tensor, sample_rate: int
    ) -> Tuple[torch.Tensor, int]:
        total_samples = tensor.shape[0]

        mu_samples = int(self.mu * sample_rate)
        sigma_samples = int(np.sqrt(self.sigma2) * sample_rate)
        segments = []
        start = 0

        while start < total_samples:
            segment_duration = max(
                int(torch.normal(mu_samples, sigma_samples).item()), 1
            )
            end = min(start + segment_duration, total_samples)
            segment = tensor[start:end]

            if segment.abs().mean().item() >= self.amplitude_threshold:
                segments.append(segment)

            start = end

        permuted_segments = segments.copy()
        permuted_indices = torch.randperm(len(permuted_segments))
        permuted_segments = [permuted_segments[i] for i in permuted_indices]

        transformed_tensor = torch.cat(permuted_segments)

        return transformed_tensor, sample_rate


class Noise(Augmentation):
    """
    This class implements an additive noise augmentation for audio data,
    simulating a desired Signal-to-Noise Ratio (SNR).

    This augmentation randomly selects a noise segment from a predefined
    `SegmentSet`, and mixes it with the input audio at a random SNR level
    within the provided range. If necessary, the noise is repeated or
    trimmed to match the length of the input signal.

    Arguments
    ---------
    segs : SegmentSet
        A collection of noise audio segments to sample from.
    snr_range : List[int]
        A range of possible SNR values in dB (e.g., [5, 20]).

    Example
    -------
    >>> segs = SegmentSet("/path/to/noise_segments")
    >>> noise_aug = Noise(segs=segs, snr_range=[10, 20])
    >>> waveform = torch.randn(16000)  # 1 second of dummy audio at 16kHz
    >>> augmented, sr = noise_aug(waveform, 16000)
    >>> augmented.shape
    torch.Size([16000])
    """

    def __init__(self, segs: SegmentSet, snr_range: List[int]) -> None:
        self.segs = segs
        # self.segs.load_audio()
        self.snr_range = snr_range

    def __call__(
        self, tensor: torch.Tensor, sample_rate: int
    ) -> Tuple[torch.Tensor, int]:
        snr_db = random.randint(self.snr_range[0], self.snr_range[1])
        rnd = self.segs.get_random()
        noise_tensor, noise_sample_rate = rnd.load_audio()

        if len(noise_tensor) > len(tensor):
            start = random.randint(0, len(noise_tensor) - len(tensor))
            noise_tensor = noise_tensor[start : start + len(tensor)]
        else:
            n = math.ceil(len(tensor) / len(noise_tensor))
            noise_tensor = noise_tensor.repeat(n)
            start = random.randint(0, len(noise_tensor) - len(tensor))
            noise_tensor = noise_tensor[start : start + len(tensor)]

        speech_power = torch.linalg.vector_norm(tensor, ord=2) ** 2
        noise_power = torch.linalg.vector_norm(noise_tensor, ord=2) ** 2 + 1e-10

        original_snr_db = 10 * (torch.log10(speech_power) - torch.log10(noise_power))
        scale = 10 ** ((original_snr_db - snr_db) / 20.0)
        tensor = scale * noise_tensor + tensor

        if torch.isnan(tensor).any():
            print("ERRREUR")
            print(rnd.segmentid)

        return tensor, sample_rate


class Codec16kHzWithEffector(Augmentation):
    def __init__(
        self,
        mp3_compression_range=(-9, 0),
        vorbis_compression_range=(-1, 10),
        opus_compression_range=(-1, 10),
        aac_compression_range=(64, 256),
        wma_compression_range=(48, 192),
    ):
        self.mp3_compression_range = mp3_compression_range
        self.vorbis_compression_range = vorbis_compression_range
        self.opus_compression_range = opus_compression_range
        self.aac_compression_range = aac_compression_range
        self.wma_compression_range = wma_compression_range

        self.codec = [
            (
                {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
                "8-bit mu-law",
            ),
            (
                {"format": "wav", "encoding": "ALAW", "bits_per_sample": 8},
                "8-bit a-law",
            ),
            ({"format": "mp3", "compression": None}, "MP3"),
            ({"format": "vorbis", "compression": None}, "Vorbis"),
            ({"format": "flac", "compression": 5}, "FLAC"),
            # ({"format": "opus", "compression": None}, "Opus"),
            # ({"format": "aac", "audio_bitrate": None}, "AAC"),
            # ({"format": "wma", "audio_bitrate": None}, "WMA")
        ]

    def __call__(self, tensor: torch.Tensor, sample_rate: int = 16000):
        param, title = random.choice(self.codec)

        if title == "MP3":
            param["compression"] = random.uniform(*self.mp3_compression_range)
        elif title == "Vorbis":
            param["compression"] = random.uniform(*self.vorbis_compression_range)
        elif title == "Opus":
            param["compression"] = random.uniform(*self.opus_compression_range)
        elif title == "AAC":
            param["audio_bitrate"] = (
                random.randint(*self.aac_compression_range) * 1000
            )  # Convert to bps
        elif title == "WMA":
            param["audio_bitrate"] = (
                random.randint(*self.wma_compression_range) * 1000
            )  # Convert to bps

        """
        effector = torchaudio.io.AudioEffector()
        effector.set_output_format(param["format"])  # Set the output format (e.g., 'mp3', 'aac')
        
        if "bitrate" in param:
            effector.set_codec_bitrate(param["bitrate"])
        
        if "compression" in param:
            effector.set_codec_compression(param["compression"])
        
        if "bits_per_sample" in param:
            ffector.set_bits_per_sample(param["bits_per_sample"])
        
        speech = effector.apply(tensor.unsqueeze(0), sample_rate)
        """

        speech = torchaudio.functional.apply_codec(
            tensor.unsqueeze(0), sample_rate, **param
        )

        return speech[0], sample_rate


class Codec(Augmentation):
    def __init__(self):
        self.codec = [
            (
                {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
                "8 bit mu-law",
            ),
            (
                {"format": "wav", "encoding": "ALAW", "bits_per_sample": 8},
                "8 bit a-law",
            ),
            ({"format": "mp3", "compression": -9}, "MP3"),
            ({"format": "vorbis", "compression": -1}, "Vorbis"),
            ({"format": "mp3", "compression": -9}, "MP3"),
            ({"format": "vorbis", "compression": -1}, "Vorbis"),
        ]

        # {"format": "mp3", "codec_config": CodecConfig(compression_level=9)},
        # {"format": "ogg", "encoder": "vorbis"},
        #

    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        param, title = random.choice(self.codec)
        speech = torchaudio.functional.apply_codec(
            tensor.unsqueeze(0), sample_rate, **param
        )

        return speech[0], sample_rate


class MediaG722(Augmentation):
    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        encoder = torchaudio.io.AudioEffector(format="g722")
        waveform = encoder.apply(tensor.unsqueeze(1), sample_rate=sample_rate)
        return waveform.squeeze(1).unsqueeze(0)[0], sample_rate


class MediaVorbis(Augmentation):
    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        r = random.randint(6, 9)
        encoder = torchaudio.io.AudioEffector(
            format="ogg",
            encoder="vorbis",
            codec_config=torchaudio.io.CodecConfig(compression_level=r),
        )
        waveform = encoder.apply(tensor.unsqueeze(1), sample_rate=sample_rate)
        return waveform.squeeze(1).unsqueeze(0)[0], sample_rate


class MediaOpus(Augmentation):
    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        r = random.randint(6, 9)
        encoder = torchaudio.io.AudioEffector(
            format="ogg",
            encoder="opus",
            codec_config=torchaudio.io.CodecConfig(compression_level=r),
        )
        waveform = encoder.apply(tensor.unsqueeze(1), sample_rate=sample_rate)
        return waveform.squeeze(1).unsqueeze(0)[0], sample_rate


class MediaMP3(Augmentation):
    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        r = random.randint(0, 1)
        if r == 0:
            # MP3 with constant bitrate
            b = random.randint(128000, 3200000)
            encoder = torchaudio.io.AudioEffector(
                format="mp3", codec_config=torchaudio.io.CodecConfig(bit_rate=b)
            )
            waveform = encoder.apply(tensor.unsqueeze(1), sample_rate=sample_rate)
            return waveform.squeeze(1).unsqueeze(0)[0], sample_rate

        else:
            # MP3 with variable bitrate
            q = random.randint(0, 3)
            c = random.randint(6, 9)
            encoder = torchaudio.io.AudioEffector(
                format="mp3",
                codec_config=torchaudio.io.CodecConfig(qscale=q, compression_level=c),
            )
            waveform = encoder.apply(tensor.unsqueeze(1), sample_rate=sample_rate)
            return waveform.squeeze(1).unsqueeze(0)[0], sample_rate


class SignFlip(Augmentation):
    """
    This class implements a simple data augmentation technique that randomly
    flips the sign of a given tensor based on a specified probability.

    It is commonly used in audio and signal processing tasks to improve
    the robustness of models by introducing variability in the training data.

    Reference: CADDA: Class-wise Automatic Differentiable Data Augmentation for EEG Signals
    Rommel et al., https://arxiv.org/abs/2106.13695

    Arguments
    ---------
    flip_prob : float
        Probability of flipping the sign of the input tensor (default: 0.5).

    Example
    -------
    >>> tensor = torch.tensor([1.0, -2.0, 3.0])
    >>> sample_rate = 16000
    >>> aug = SignFlip(flip_prob=1.0)
    >>> flipped_tensor, sr = aug(tensor, sample_rate)
    >>> flipped_tensor
    tensor([-1.0, 2.0, -3.0])
    >>> sr
    16000
    """

    def __init__(self, flip_prob: float = 0.5) -> None:
        self.flip_prob = flip_prob

    def __call__(
        self, tensor: torch.Tensor, sample_rate: int
    ) -> Tuple[torch.Tensor, int]:
        if torch.rand(1).item() < self.flip_prob:
            return -tensor, sample_rate

        return tensor, sample_rate


class Reverb(Augmentation):
    """
    This class applies reverberation-based audio augmentation using
    Room Impulse Response (RIR) convolution.

    The augmentation convolves input audio with a randomly selected
    RIR from a provided segment set, simulating reverberant environments
    for robust audio processing tasks.

    Arguments
    ---------
    segs : SegmentSet
        A set of audio segments (RIRs) from which one is randomly chosen
        to apply reverberation.

    Example
    -------
    >>> segs = SegmentSet(...)
    >>> reverb_aug = Reverb(segs)
    >>> audio_tensor = torch.rand(16000)
    >>> sample_rate = 16000
    >>> augmented_audio, new_sr = reverb_aug(audio_tensor, sample_rate)
    >>> augmented_audio.shape
    torch.Size([16000])
    """

    def __init__(self, segs: SegmentSet) -> None:
        self.segs = segs
        self.rir_scaling_factor = 0.5**15

    def __call__(
        self, tensor: torch.Tensor, sample_rate: int
    ) -> Tuple[torch.Tensor, int]:
        reverb_tensor, reverb_sample_rate = self.segs.get_random().load_audio()
        size = len(tensor)

        power_before = torch.dot(tensor, tensor) / len(tensor)

        tensor = torch.Tensor(
            signal.convolve(tensor, reverb_tensor, mode="full")[:size]
        )
        power_after = torch.dot(tensor, tensor) / size
        tensor *= (power_before / power_after).sqrt()
        tensor = torch.clamp(tensor, -1.0, 1.0)

        return tensor, sample_rate


class Filtering(Augmentation):
    def __init__(self):
        pass

    def __call__(self, tensor: torch.Tensor, sample_rate: int):
        effects = [["bandpass", "3400", "3700"], ["bandreject", "100", "300"]]
        e = effects[random.randint(0, 1)]

        speech, _ = torchaudio.sox_effects.apply_effects_tensor(
            tensor.unsqueeze(0), sample_rate, effects=[e]
        )

        return speech[0], sample_rate


class VAD(Augmentation):
    """
    This class implements a simple Voice Activity Detection (VAD) module
    based on energy thresholding.

    The algorithm determines which frames of an audio signal contain speech
    by comparing the log energy to a dynamic threshold. The decision for each
    frame is made using the energy of neighboring frames and a proportion
    threshold.

    Reference: The Kaldi speech recognition toolkit
    Povey et al., https://github.com/kaldi-asr/kaldi/blob/master/src/ivector/voice-activity-detection.cc

    Arguments
    ---------
    vad_energy_threshold : float
        Base energy threshold below which a frame is considered silent (default: -12.0).
    vad_energy_mean_scale : float
        Factor to adjust threshold based on mean energy of the input (default: 0.3).
    vad_frames_context : int
        Number of frames before and after to consider in the local window (default: 2).
    vad_proportion_threshold : float
        Minimum proportion of frames in the window that must be below threshold
        to classify a frame as non-speech (default: 0.3).

    Example
    -------
    >>> tensor = torch.randn(100, 1)  # Simulated log-energy signal
    >>> vad = VAD()
    >>> voiced_tensor = vad(tensor)
    >>> voiced_tensor.shape
    torch.Size([N, 1])  # N <= 100, depending on detected speech frames
    """

    def __init__(
        self,
        vad_energy_threshold: float = -12.0,
        vad_energy_mean_scale: float = 0.3,
        vad_frames_context: int = 2,
        vad_proportion_threshold: float = 0.3,
    ) -> None:
        self.vad_energy_threshold = vad_energy_threshold
        self.vad_energy_mean_scale = vad_energy_mean_scale
        self.vad_frames_context = vad_frames_context
        self.vad_proportion_threshold = vad_proportion_threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        T = tensor.size(0)
        output_voiced = torch.zeros(T)

        log_energy = tensor[:, 0]

        energy_threshold = self.vad_energy_threshold
        if self.vad_energy_mean_scale != 0.0:
            energy_threshold -= self.vad_energy_mean_scale * log_energy.sum() / T

        for t in range(T):
            num_count = 0
            den_count = 0

            for t2 in range(
                max(0, t - self.vad_frames_context),
                min(T, t + self.vad_frames_context + 1),
            ):
                den_count += 1
                if log_energy[t2].item() < energy_threshold:
                    num_count += 1

            if num_count >= den_count * self.vad_proportion_threshold:
                output_voiced[t] = 0.0
            else:
                output_voiced[t] = 1.0

        return tensor[output_voiced.bool()]


class Crop(Augmentation):
    """
    This class implements a time-based cropping augmentation for input tensors.

    The augmentation extracts a segment of a fixed duration from the input tensor,
    either randomly or deterministically from the beginning. If the input tensor is
    shorter than the desired duration, it is repeated to meet the required length.

    Arguments
    ---------
    duration : int
        Target duration (number of time steps) to crop from the input tensor.
    random : bool, optional
        If True, crop a random segment from the tensor. If False, crop from the beginning.
        Default is True.

    Example
    -------
    >>> tensor = torch.rand([100, 80])
    >>> aug = Crop(duration=50)
    >>> out_tensor = aug(tensor)
    >>> out_tensor.shape
    torch.Size([50, 80])
    """

    def __init__(self, duration: int, random: bool = True) -> None:
        self.duration = duration
        self.random = random

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.random == True:
            if tensor.shape[0] < self.duration:
                n = math.ceil(self.duration / tensor.shape[0])
                tensor = tensor.repeat(n, 1)

            max_start_time = tensor.shape[0] - self.duration
            start_time = random.randint(0, max_start_time)
            result = tensor[start_time : start_time + self.duration, :]
            return result
        else:
            max_start_time = self.duration

            if tensor.shape[0] < self.duration:
                max_start_time = tensor.shape[0]

            result = tensor[0:max_start_time, :]
            return result


class SpecAugment(Augmentation):
    """
    This class implements the SpecAugment data augmentation technique
    for audio spectrograms.

    SpecAugment applies random time and frequency masking to the input
    spectrogram to improve model robustness and generalization, particularly
    in speech recognition tasks.

    Reference: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,
    Park et al., https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    num_t_mask : int
        Number of time masks to apply.
    num_f_mask : int
        Number of frequency masks to apply.
    max_t : int
        Maximum width of each time mask (in frames).
    max_f : int
        Maximum width of each frequency mask (in bins).
    prob : float
        Probability of applying augmentation (default: 0.6).

    Example
    -------
    >>> spectrogram = torch.randn([100, 80])  # [time, frequency]
    >>> aug = SpecAugment(num_t_mask=2, num_f_mask=2, max_t=20, max_f=10, prob=1.0)
    >>> augmented = aug(spectrogram)
    >>> augmented.shape
    torch.Size([100, 80])
    """

    def __init__(
        self,
        num_t_mask: int = 1,
        num_f_mask: int = 1,
        max_t: int = 10,
        max_f: int = 8,
        prob: float = 0.6,
    ) -> None:
        self.num_t_mask = num_t_mask
        self.num_f_mask = num_f_mask
        self.max_t = max_t
        self.max_f = max_f
        self.prob = prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
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
    """
    This class implements Cepstral Mean and Variance Normalization (CMVN)
    as an audio/data augmentation technique.

    CMVN is commonly used to normalize features like Mel-spectrograms or
    MFCCs by removing mean (and optionally variance) across time, which
    helps reduce speaker and channel variability.

    Arguments
    ---------
    norm_means : bool
        If True, subtracts the mean across time dimension (default: True).
    norm_vars : bool
        If True, normalizes by the standard deviation (not implemented here) (default: False).

    Example
    -------
    >>> cmvn = CMVN(norm_means=True, norm_vars=False)
    >>> input_tensor = torch.rand([100, 40])  # 100 frames, 40 features
    >>> output_tensor = cmvn(input_tensor)
    >>> output_tensor.shape
    torch.Size([100, 40])
    """

    def __init__(self, norm_means: bool = True, norm_vars: bool = False) -> None:
        self.norm_means = norm_means
        self.norm_vars = norm_vars

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.norm_means == True:
            tensor = tensor - tensor.mean(dim=0)
        if self.norm_vars:
            std = tensor.std(dim=0).clamp(min=1e-8)
            tensor = tensor / std
        return tensor
