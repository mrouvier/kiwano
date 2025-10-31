kiwano/model/automodel.py import torch

from kiwano.model import KiwanoResNet, ResNet, XIKiwanoResNet


class AutoModel(torch.nn.Module):
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, ckpt):
        extra_repr = ckpt["extra_repr"]
        pairs = extra_repr.split(", ")
        result_dict = {}
        for pair in pairs:
            key, value = pair.split("=")
            result_dict[key] = int(value)

        model = None

        if ckpt["name"] == "ResNet":
            model = ResNet(
                num_classes=result_dict["num_classes"],
                embed_features=result_dict["embed_features"],
            )

        if ckpt["name"] == "KiwanoResNet":
            model = KiwanoResNet(
                num_classes=result_dict["num_classes"],
                embed_features=result_dict["embed_features"],
            )

        if ckpt["name"] == "XIKiwanoResNet":
            model = XIKiwanoResNet(
                num_classes=result_dict["num_classes"],
                embed_features=result_dict["embed_features"],
            )

        model.load_state_dict(ckpt["model"])

        return model
