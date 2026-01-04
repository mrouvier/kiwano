#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kiwano.augmentation import (
    CMVN,
    Augmentation,
    Codec,
    Compose,
    Crop,
    Filtering,
    Noise,
    Normal,
    OneOf,
    Reverb,
    SpecAugment,
)
from kiwano.dataset import Segment, SegmentSet
from kiwano.features import Fbank
from kiwano.model import JeffreysLoss, KiwanoResNet, WarmupPlateauScheduler
from kiwano.utils import Pathlike

logger = logging.getLogger(__name__)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    def __init__(
        self,
        audio_transforms: List[Augmentation] = None,
        feature_extractor=None,
        feature_transforms: List[Augmentation] = None,
    ):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        segment = None
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(
                val
                for idx, val in enumerate(self.segments.values())
                if idx == segment_id_or_index
            )

        audio, sample_rate = segment.load_audio()
        if self.audio_transforms != None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)

        if self.feature_extractor != None:
            feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)

        if self.feature_transforms != None:
            feature = self.feature_transforms(feature)

        return feature, self.labels[segment.spkid]


if __name__ == "__main__":

    NODE_ID = os.environ["SLURM_NODEID"]
    MASTER_ADDR = os.environ["MASTER_ADDR"]

    accelerator = Accelerator(mixed_precision="fp16")  # or "no", "bf16" as you like

    if accelerator.is_main_process:
        print(
            ">>> Training on ",
            accelerator.num_processes,
            " processes, master node is ",
            MASTER_ADDR,
        )

    checkpoint = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--musan", type=str, default="data/musan/")
    parser.add_argument("--rirs_noises", type=str, default="data/rirs_noises/")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--max_epochs", type=int, default=51)
    parser.add_argument("training_corpus", type=str, metavar="training_corpus")
    parser.add_argument("exp_dir", type=str, metavar="exp_dir")

    args = parser.parse_args()

    print("#" + " ".join(sys.argv[0:]))
    print("# Started at " + time.ctime())
    print("#")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))

    epochs_start = 0
    if args.checkpoint:
        epochs_start = checkpoint["epochs"]

    musan = SegmentSet()
    musan.from_dict(Path(args.musan))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    reverb = SegmentSet()
    reverb.from_dict(Path(args.rirs_noises))

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=OneOf(
            [
                Noise(musan_music, snr_range=[5, 15]),
                Noise(musan_speech, snr_range=[13, 20]),
                Noise(musan_noise, snr_range=[0, 15]),
                Normal(),
                Reverb(reverb),
            ]
        ),
        feature_extractor=Fbank(),
        feature_transforms=Compose(
            [
                CMVN(),
                Crop(350),
                SpecAugment(),
            ]
        ),
    )

    training_data.from_dict(Path(args.training_corpus))

    training_data.describe()

    train_dataloader = DataLoader(
        training_data,
        batch_size=32,
        drop_last=True,
        shuffle=True,
        num_workers=15,
        pin_memory=True,
    )
    iterator = iter(train_dataloader)

    resnet_model = KiwanoResNet()
    if args.checkpoint:
        resnet_model.load_state_dict(checkpoint["model"])
    resnet_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)

    optimizer = torch.optim.SGD(
        [
            {
                "params": resnet_model.preresnet.parameters(),
                "weight_decay": 0.0001,
                "lr": 0.00001,
            },
            {
                "params": resnet_model.temporal_pooling.parameters(),
                "weight_decay": 0.0001,
                "lr": 0.00001,
            },
            {
                "params": resnet_model.embedding.parameters(),
                "weight_decay": 0.0001,
                "lr": 0.00001,
            },
            {"params": resnet_model.output.parameters(), "lr": 0.00001},
        ],
        momentum=0.9,
    )

    if args.checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    resnet_model, optimizer, train_dataloader = accelerator.prepare(
        resnet_model, optimizer, train_dataloader
    )

    criterion = JeffreysLoss(coeff1=0.1, coeff2=0.025)

    scheduler = WarmupPlateauScheduler(
        optimizer,
        max_epochs=args.max_epochs,
        initial_lr=0.2,
        warm_up_epoch=5,
        plateau_epoch=15,
        patience=10,
        factor=5,
        margin_loss=0.3,
    )
    if args.checkpoint:
        scheduler.set_epoch(checkpoint["epochs"])

    running_loss = [np.nan for _ in range(500)]

    for epochs in range(epochs_start, args.max_epochs):
        iterations = 0
        unwrapped_model = accelerator.unwrap_model(resnet_model)
        unwrapped_model.set_m(scheduler.get_margin_loss())

        accelerator.wait_for_everyone()

        for feats, iden in train_dataloader:

            feats = feats.unsqueeze(1)
            feats = feats.float().to(accelerator.device, non_blocking=True)
            iden = iden.to(accelerator.device, non_blocking=True)

            optimizer.zero_grad()

            with accelerator.autocast():
                preds = resnet_model(feats, iden)
                loss = criterion(preds, iden)

            accelerator.backward(loss)
            optimizer.step()

            running_loss.pop(0)
            running_loss.append(loss.item())
            rmean_loss = np.nanmean(np.array(running_loss))

            if iterations % 100 == 0:
                msg = "{}: Epoch: [{}/{}] ({}/{}) \t AvgLoss:{:.4f} \t C-Loss:{:.4f} \t LR : {:.8f} \t Margin : {:.4f}".format(
                    time.ctime(),
                    epochs,
                    150,
                    iterations,
                    len(train_dataloader),
                    rmean_loss,
                    loss.item(),
                    get_lr(optimizer),
                    unwrapped_model.get_m(),
                )
                print(msg)

            iterations += 1

        scheduler.step()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            checkpoint = {
                "epochs": epochs + 1,
                "optimizer": optimizer.state_dict(),
                "model": unwrapped_model.state_dict(),
                "name": type(unwrapped_model).__name__,
                "config": unwrapped_model.extra_repr(),
            }
            accelerator.save(
                checkpoint, args.exp_dir + "/model" + str(epochs) + ".ckpt"
            )
