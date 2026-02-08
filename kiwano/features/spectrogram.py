from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio


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


class FeatureExtractor(metaclass=ABCMeta):
    name = None
    config_type = None

    def __init__(self, config: Optional[Any] = None):
        if config is None:
            config = self.config_type()
        self.config = config

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        pass


@dataclass
class SpectrogramConfig:
    sampling_rate: int = 16000

    def to_dict(self) -> Dict[str, Any]:
        return asdict_nonull(self)

    def from_dict(data: Dict[str, Any]) -> "SpectrogramConfig":
        return SpectrogramConfig(**data)


class Spectrogram(FeatureExtractor):
    name = "spectrogram"
    config_type = SpectrogramConfig

    def __init__(self, config: Optional[SpectrogramConfig] = None):
        super().__init__(config=config)

        self.extractor = torchaudio.transforms.Spectrogram(
            n_fft=511,
            win_length=400,
            hop_length=160,
            pad=0,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
            window_fn=torch.hamming_window,
        )

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Spectrogram was instantiated for sampling_rate "
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

        feats = torch.log(torch.add(feats, 9.9999999999999995e-07))

        feats = feats - torch.mean(feats, [-1], True)

        feats = feats.transpose(0, 1)

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats
