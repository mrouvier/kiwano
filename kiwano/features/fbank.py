import math
import warnings

import numpy as np
import torch
from torch import nn
from typing import Any, Dict, Optional, Union, List, Tuple
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from torch.fft import rfft as torch_rfft

HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"

EPSILON = 1e-10

def asdict_nonull(dclass) -> Dict[str, Any]:
    """
    Recursively convert a dataclass into a dict, removing all the fields with `None` value.
    Intended to use in place of dataclasses.asdict(), when the null values are not desired in the serialized document.
    """

    def non_null_dict_factory(collection):
        d = dict(collection)
        remove_keys = []
        for key, val in d.items():
            if val is None:
                remove_keys.append(key)
        for k in remove_keys:
            del d[k]
        return d

    return asdict(dclass, dict_factory=non_null_dict_factory)





def _rfft(x: torch.Tensor) -> torch.Tensor:
    return torch_rfft(x, dim=-1)

def _pow_spectrogram(x: torch.Tensor) -> torch.Tensor:
    return x.abs() ** 2

def _spectrogram(x: torch.Tensor) -> torch.Tensor:
    return x.abs()



def _get_log_energy(x: torch.Tensor, energy_floor: float) -> torch.Tensor:
    """
    Returns the log energy of size (m) for a strided_input (m,*)
    """
    log_energy = (x.pow(2).sum(-1) + 1e-15).log()  # size (m)
    if energy_floor > 0.0:
        log_energy = torch.max(
            log_energy,
            torch.tensor(math.log(energy_floor), dtype=log_energy.dtype),
        )

    return log_energy


def next_power_of_2(x: int) -> int:
    """
    Returns the smallest power of 2 that is greater than x.
    Original source: TorchAudio (torchaudio/compliance/kaldi.py)
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided_batch(
    waveform: torch.Tensor, window_length: int, window_shift: int, snip_edges: bool
) -> torch.Tensor:
    r"""Given a waveform (2D tensor of size ``(batch_size, num_samples)``,
    it returns a 2D tensor ``(batch_size, num_frames, window_length)``
    representing how the window is shifted along the waveform. Each row is a frame.
    Args:
        waveform (torch.Tensor): Tensor of size ``(batch_size, num_samples)``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.
    Returns:
        torch.Tensor: 3D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert waveform.dim() == 2
    batch_size = waveform.size(0)
    num_samples = waveform.size(-1)

    if snip_edges:
        if num_samples < window_length:
            return torch.empty((0, 0, 0))
        else:
            num_frames = 1 + (num_samples - window_length) // window_shift
    else:
        num_frames = (num_samples + (window_shift // 2)) // window_shift
        new_num_samples = (num_frames - 1) * window_shift + window_length
        npad = new_num_samples - num_samples
        npad_left = int((window_length - window_shift) // 2)
        npad_right = npad - npad_left
        # waveform = nn.functional.pad(waveform, (npad_left, npad_right), mode='reflect')
        pad_left = torch.flip(waveform[:, :npad_left], (1,))
        if npad_right >= 0:
            pad_right = torch.flip(waveform[:, -npad_right:], (1,))
        else:
            pad_right = torch.zeros(0, dtype=waveform.dtype)
        waveform = torch.cat((pad_left, waveform, pad_right), dim=1)

    strides = (
        waveform.stride(0),
        window_shift * waveform.stride(1),
        waveform.stride(1),
    )
    sizes = [batch_size, num_frames, window_length]
    return waveform.as_strided(sizes, strides)



def create_frame_window(window_size, window_type: str = "povey", blackman_coeff=0.42):
    r"""Returns a window function with the given type and size"""
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46)
    elif window_type == POVEY:
        return torch.hann_window(window_size, periodic=False).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, dtype=torch.get_default_dtype())
    elif window_type == BLACKMAN:
        a = 2 * math.pi / window_size
        window_function = torch.arange(window_size, dtype=torch.get_default_dtype())
        return (
            blackman_coeff
            - 0.5 * torch.cos(a * window_function)
            + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
        )
    else:
        raise Exception(f"Invalid window type: {window_type}")



class Wav2Win(nn.Module):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and partition them into overlapping frames (of audio samples).
    Note: no feature extraction happens in here, the output is still a time-domain signal.
    Example::
        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2Win()
        >>> t(x).shape
        torch.Size([1, 100, 400])
    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, window_length)``.
    When ``return_log_energy==True``, returns a tuple where the second element
    is a log-energy tensor of shape ``(batch_size, num_frames)``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: float = 0.025,
        frame_shift: float = 0.01,
        pad_length: Optional[int] = None,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = EPSILON,
        raw_energy: bool = True,
        return_log_energy: bool = False,
    ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.remove_dc_offset = remove_dc_offset
        self.preemph_coeff = preemph_coeff
        self.window_type = window_type
        self.dither = dither
        # torchscript expects it to be a tensor
        self.snip_edges = snip_edges
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.return_log_energy = return_log_energy
        if snip_edges:
            warnings.warn(
                "Setting snip_edges=True is generally incompatible with Lhotse -- "
                "you might experience mismatched duration/num_frames errors."
            )

        N = int(math.floor(frame_length * sampling_rate))
        self._length = N
        self._shift = int(math.floor(frame_shift * sampling_rate))

        self._window = nn.Parameter(
            create_frame_window(N, window_type=window_type), requires_grad=False
        )
        self.pad_length = N if pad_length is None else pad_length
        assert (
            self.pad_length >= N
        ), f"pad_length (or fft_length) = {pad_length} cannot be smaller than N = {N}"

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = (
            "{}(sampling_rate={}, frame_length={}, frame_shift={}, pad_length={}, "
            "remove_dc_offset={}, preemph_coeff={}, window_type={} "
            "dither={}, snip_edges={}, energy_floor={}, raw_energy={}, return_log_energy={})"
        ).format(
            self.__class__.__name__,
            self.sampling_rate,
            self.frame_length,
            self.frame_shift,
            self.pad_length,
            self.remove_dc_offset,
            self.preemph_coeff,
            self.window_type,
            self.dither,
            self.snip_edges,
            self.energy_floor,
            self.raw_energy,
            self.return_log_energy,
        )
        return s

    def _forward_strided(
        self, x_strided: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # remove offset
        if self.remove_dc_offset:
            mu = torch.mean(x_strided, dim=2, keepdim=True)
            x_strided = x_strided - mu

        # Compute the log energy of each frame
        log_energy: Optional[torch.Tensor] = None
        if self.return_log_energy and self.raw_energy:
            log_energy = _get_log_energy(x_strided, self.energy_floor)  # size (m)

        # preemphasis
        if self.preemph_coeff != 0.0:
            x_offset = torch.nn.functional.pad(x_strided, (1, 0), mode="replicate")
            x_strided = x_strided - self.preemph_coeff * x_offset[:, :, :-1]

        # Apply window_function to each frame
        x_strided = x_strided * self._window

        # Pad columns with zero until we reach size (batch, num_frames, pad_length)
        if self.pad_length != self._length:
            pad = self.pad_length - self._length
            x_strided = torch.nn.functional.pad(
                # torchscript expects pad to be list of int
                x_strided.unsqueeze(1),
                [0, pad],
                mode="constant",
                value=0.0,
            ).squeeze(1)

        if self.return_log_energy and not self.raw_energy:
            # This energy is computed after preemphasis, window, etc.
            log_energy = _get_log_energy(x_strided, self.energy_floor)  # size (m)

        return x_strided, log_energy

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Add dither
        if self.dither != 0.0:
            n = torch.randn(x.shape, device=x.device)
            x = x + self.dither * n

        x_strided = _get_strided_batch(x, self._length, self._shift, self.snip_edges)

        return self._forward_strided(x_strided)




class Wav2FFT(nn.Module):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their Short-Time Fourier Transform (STFT).
    The output is a complex-valued tensor.
    Example::
        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2FFT()
        >>> t(x).shape
        torch.Size([1, 100, 257])
    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_fft_bins)``
    with dtype ``torch.complex64``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: float = 0.025,
        frame_shift: float = 0.01,
        round_to_power_of_two: bool = True,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = EPSILON,
        raw_energy: bool = True,
        use_energy: bool = True,
    ) -> None:
        super().__init__()
        self.use_energy = use_energy
        N = int(math.floor(frame_length * sampling_rate))
        self.fft_length = next_power_of_2(N) if round_to_power_of_two else N
        self.wav2win = Wav2Win(
            sampling_rate,
            frame_length,
            frame_shift,
            pad_length=self.fft_length,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            return_log_energy=use_energy,
        )

    @property
    def sampling_rate(self) -> int:
        return self.wav2win.sampling_rate

    @property
    def frame_length(self) -> float:
        return self.wav2win.frame_length

    @property
    def frame_shift(self) -> float:
        return self.wav2win.frame_shift

    @property
    def remove_dc_offset(self) -> bool:
        return self.wav2win.remove_dc_offset

    @property
    def preemph_coeff(self) -> float:
        return self.wav2win.preemph_coeff

    @property
    def window_type(self) -> str:
        return self.wav2win.window_type

    @property
    def dither(self) -> float:
        return self.wav2win.dither

    def _forward_strided(
        self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Note: subclasses of this module can override ``_forward_strided()`` and get a working
        # implementation of ``forward()`` and ``online_inference()`` for free.
        X = _rfft(x_strided)

        # log_e is not None is needed by torchscript
        if self.use_energy and log_e is not None:
            X[:, :, 0] = log_e

        return X

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_strided, log_e = self.wav2win(x)
        return self._forward_strided(x_strided=x_strided, log_e=log_e)




class Wav2LogFilterBank(Wav2FFT):
    """
    Apply standard Kaldi preprocessing (dithering, removing DC offset, pre-emphasis, etc.)
    on the input waveforms and compute their log-Mel filter bank energies (also known as "fbank").
    Example::
        >>> x = torch.randn(1, 16000, dtype=torch.float32)
        >>> x.shape
        torch.Size([1, 16000])
        >>> t = Wav2LogFilterBank()
        >>> t(x).shape
        torch.Size([1, 100, 80])
    The input is a tensor of shape ``(batch_size, num_samples)``.
    The output is a tensor of shape ``(batch_size, num_frames, num_filters)``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frame_length: float = 0.025,
        frame_shift: float = 0.01,
        round_to_power_of_two: bool = True,
        remove_dc_offset: bool = True,
        preemph_coeff: float = 0.97,
        window_type: str = "povey",
        dither: float = 0.0,
        snip_edges: bool = False,
        energy_floor: float = EPSILON,
        raw_energy: bool = True,
        use_energy: bool = False,
        use_fft_mag: bool = False,
        low_freq: float = 20.0,
        high_freq: float = -400.0,
        num_filters: int = 80,
        norm_filters: bool = False,
        torchaudio_compatible_mel_scale: bool = True,
    ):

        super().__init__(
            sampling_rate,
            frame_length,
            frame_shift,
            round_to_power_of_two=round_to_power_of_two,
            remove_dc_offset=remove_dc_offset,
            preemph_coeff=preemph_coeff,
            window_type=window_type,
            dither=dither,
            snip_edges=snip_edges,
            energy_floor=energy_floor,
            raw_energy=raw_energy,
            use_energy=use_energy,
        )

        self.use_fft_mag = use_fft_mag
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_filters = num_filters
        self.norm_filters = norm_filters
        self._eps = nn.Parameter(
            torch.tensor(torch.finfo(torch.float).eps), requires_grad=False
        )

        if use_fft_mag:
            self._to_spec = _spectrogram
        else:
            self._to_spec = _pow_spectrogram

        if torchaudio_compatible_mel_scale:
            from torchaudio.compliance.kaldi import get_mel_banks

            # see torchaudio.compliance.kaldi.fbank, lines #581-587 for the original usage
            fb, _ = get_mel_banks(
                num_bins=num_filters,
                window_length_padded=self.fft_length,
                sample_freq=sampling_rate,
                low_freq=low_freq,
                high_freq=high_freq,
                # VTLN args are hardcoded to torchaudio default values;
                # they are not used anyway with wapr_factor == 1.0
                vtln_warp_factor=1.0,
                vtln_low=100.0,
                vtln_high=-500.0,
            )
            fb = torch.nn.functional.pad(fb, (0, 1), mode="constant", value=0).T
        else:
            fb = create_mel_scale(
                num_filters=num_filters,
                fft_length=self.fft_length,
                sampling_rate=sampling_rate,
                low_freq=low_freq,
                high_freq=high_freq,
                norm_filters=norm_filters,
            )
        self._fb = nn.Parameter(fb, requires_grad=False)

    def _forward_strided(
        self, x_strided: torch.Tensor, log_e: Optional[torch.Tensor]
    ) -> torch.Tensor:
        X = _rfft(x_strided)
        pow_spec = self._to_spec(X)

        pow_spec = torch.matmul(pow_spec, self._fb)
        pow_spec = torch.max(pow_spec, self._eps).log()

        # log_e is not None is needed by torchscript
        if self.use_energy and log_e is not None:
            pow_spec = torch.cat((log_e.unsqueeze(-1), pow_spec), dim=-1)

        return pow_spec




@dataclass
class FbankConfig:
    sampling_rate: int = 16000
    frame_length: float = 0.025
    frame_shift: float = 0.01
    round_to_power_of_two: bool = True
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = "povey"
    dither: float = 0.0
    snip_edges: bool = False
    energy_floor: float = EPSILON
    raw_energy: bool = True
    use_energy: bool = True
    use_fft_mag: bool = False
    low_freq: float = 20.0
    high_freq: float = -400.0
    num_filters: int = 80
    num_mel_bins: Optional[int] = None  # do not use
    norm_filters: bool = False

    def __post_init__(self):
        if self.num_mel_bins is not None:
            self.num_filters = self.num_mel_bins
            self.num_mel_bins = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict_nonull(self)

    def from_dict(data: Dict[str, Any]) -> "FbankConfig":
        return FbankConfig(**data)


class FeatureExtractor(metaclass=ABCMeta):
    name = None
    config_type = None

    def __init__(self, config: Optional[Any] = None):
        if config is None:
            config = self.config_type()
        self.config = config

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        pass

class Fbank(FeatureExtractor):

    name = "kaldi-fbank"
    config_type = FbankConfig

    def __init__(self, config: Optional[FbankConfig] = None):
        super().__init__(config=config)
        self.extractor = Wav2LogFilterBank(**self.config.to_dict()).eval()

    def frame_shift(self) -> float:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_filters

    def extract(self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Fbank was instantiated for sampling_rate "
            f"{self.config.sampling_rate}, but "
            f"sampling_rate={sampling_rate} was passed to extract()."
        )

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        feats = self.extractor(samples)[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats


