import torch

from kiwano.model import KiwanoResNet, ResNet, XIKiwanoResNet


class AutoModel(torch.nn.Module):
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, ckpt):
        repr_details = ckpt["config"]
        pairs = repr_details.split("; ")
        result_dict = {}
        for pair in pairs:
            key, value = pair.split("=")
            result_dict[key] = value

        model = None

        if ckpt["name"] == "ResNet":
            model = ResNet(
                in_channels=int(result_dict["in_channels"]),
                embed_dim=int(result_dict["embed_dim"]),
                num_classes=int(result_dict["num_classes"]),
                stage_channels=tuple(
                    map(int, result_dict["stage_channels"].split(","))
                ),
                stage_blocks=tuple(map(int, result_dict["stage_blocks"].split(","))),
                stage_strides=tuple(map(int, result_dict["stage_strides"].split(","))),
            )

        if ckpt["name"] == "KiwanoResNet":
            model = KiwanoResNet(
                in_channels=int(result_dict["in_channels"]),
                embed_dim=int(result_dict["embed_dim"]),
                num_classes=int(result_dict["num_classes"]),
                stage_channels=tuple(
                    map(int, result_dict["stage_channels"].split(","))
                ),
                stage_blocks=tuple(map(int, result_dict["stage_blocks"].split(","))),
                stage_strides=tuple(map(int, result_dict["stage_strides"].split(","))),
            )

        if ckpt["name"] == "XIKiwanoResNet":
            model = XIKiwanoResNet(
                in_channels=int(result_dict["in_channels"]),
                embed_dim=int(result_dict["embed_dim"]),
                num_classes=int(result_dict["num_classes"]),
                stage_channels=tuple(
                    map(int, result_dict["stage_channels"].split(","))
                ),
                stage_blocks=tuple(map(int, result_dict["stage_blocks"].split(","))),
                stage_strides=tuple(map(int, result_dict["stage_strides"].split(","))),
            )

        model.load_state_dict(ckpt["model"])

        return model
