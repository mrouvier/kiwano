from dataclasses import asdict

import torch

from kiwano.model import KiwanoResNet, ResNet, XIKiwanoResNet


class AutoModel(torch.nn.Module):

    _MODEL_REGISTRY = {
        "ResNet": ResNet,
        "KiwanoResNet": KiwanoResNet,
        "XIKiwanoResNet": XIKiwanoResNet,
    }

    SUPPORTED_VERSIONS = {"1.0"}

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, ckpt, map_location="cpu", device=None, strict=True):
        if isinstance(ckpt, str):
            ckpt = torch.load(ckpt, map_location=map_location)

        cls._check_version(ckpt)

        name = ckpt.get("name")
        config = ckpt.get("config")
        state_dict = ckpt.get("model")

        if name not in cls._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Available: {list(cls._MODEL_REGISTRY.keys())}"
            )

        model_cls = cls._MODEL_REGISTRY[name]

        # dataclass → dict → constructor
        #if hasattr(config, "__dataclass_fields__"):
        #    config = asdict(config)

        model = model_cls.from_config(config)
        model.load_state_dict(state_dict, strict=strict)

        if device is not None:
            model = model.to(device)

        return model

    @staticmethod
    def _check_version(ckpt):
        version = ckpt.get("version", "1.0")
        if version not in AutoModel.SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"Unsupported checkpoint version '{version}'. "
                f"Supported versions: {AutoModel.SUPPORTED_VERSIONS}"
            )
