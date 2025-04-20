#!/usr/bin/env python3


import torch

import kiwano
from kiwano.model import ResNetASVSpoof

model1 = ResNetASVSpoof(num_classes=2)
model2 = ResNetASVSpoof(num_classes=2)
model3 = ResNetASVSpoof(num_classes=2)
model4 = ResNetASVSpoof(num_classes=2)

model1.load_state_dict(
    torch.load("exp/resnet_ddp_adamw_batch256/model19.ckpt")["model"]
)
model2.load_state_dict(
    torch.load("exp/resnet_ddp_adamw_batch256/model20.ckpt")["model"]
)
model3.load_state_dict(
    torch.load("exp/resnet_ddp_adamw_batch256/model23.ckpt")["model"]
)
model4.load_state_dict(
    torch.load("exp/resnet_ddp_adamw_batch256/model26.ckpt")["model"]
)


smodel1 = model1.state_dict()
smodel2 = model2.state_dict()
smodel3 = model3.state_dict()
smodel4 = model4.state_dict()


for key in smodel1:
    smodel3[key] = (
        0.16 * smodel1[key]
        + 0.16 * smodel2[key]
        + 0.18 * smodel3[key]
        + 0.5 * smodel4[key]
    )

checkpoint = {
    "epochs": "",
    "optimizer": "",
    "model": smodel3,
    "name": type(model1).__name__,
    "config": model1.extra_repr(),
}

torch.save(checkpoint, "exp/resnet_ddp_adamw_batch256/modelaveraged.ckpt")
