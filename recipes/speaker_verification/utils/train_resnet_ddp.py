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
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

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
from kiwano.model import IDRDScheduler, JeffreysLoss, KiwanoResNet
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

    checkpoint = None

    RANK = int(os.environ["RANK"])
    LOCAL_RANK = os.environ["LOCAL_RANK"]


    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
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

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=RANK,
        output_device=LOCAL_RANK,
    )

    gpu = torch.device(f"cuda:{local_rank}")

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

    train_sampler = DistributedSampler(
        training_data,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=32,
        drop_last=True,
        shuffle=False,
        num_workers=15,
        sampler=train_sampler,
        pin_memory=True,
    )
    iterator = iter(train_dataloader)

    resnet_model = KiwanoResNet()
    if args.checkpoint:
        resnet_model.load_state_dict(checkpoint["model"])
    resnet_model.to(gpu)
    resnet_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)

    resnet_model = torch.nn.parallel.DistributedDataParallel(
        resnet_model, device_ids=[idr_torch.local_rank]
    )

    optimizer = torch.optim.SGD(
        [
            {
                "params": resnet_model.module.preresnet.parameters(),
                "weight_decay": 0.0001,
                "lr": 0.00001,
            },
            {
                "params": resnet_model.module.temporal_pooling.parameters(),
                "weight_decay": 0.0001,
                "lr": 0.00001,
            },
            {
                "params": resnet_model.module.embedding.parameters(),
                "weight_decay": 0.0001,
                "lr": 0.00001,
            },
            {"params": resnet_model.module.output.parameters(), "lr": 0.00001},
        ],
        momentum=0.9,
    )
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    criterion = JeffreysLoss(coeff1=0.1, coeff2=0.025)

    scheduler = IDRDScheduler(
        optimizer,
        num_epochs=args.max_epochs,
        initial_lr=0.2,
        warm_up_epoch=5,
        plateau_epoch=15,
        patience=10,
        factor=5,
        amsmloss=0.3,
    )
    if args.checkpoint:
        scheduler.set_epoch(checkpoint["epochs"])

    running_loss = [np.nan for _ in range(500)]

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epochs in range(epochs_start, args.max_epochs):
        iterations = 0
        train_sampler.set_epoch(epochs)
        resnet_model.module.set_m(scheduler.get_amsmloss())
        torch.distributed.barrier()
        for feats, iden in train_dataloader:

            feats = feats.unsqueeze(1)

            feats = feats.float().to(gpu)
            iden = iden.to(gpu)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                preds = resnet_model(feats, iden)
                loss = criterion(preds, iden)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
                    resnet_model.module.get_m(),
                )
                print(msg)

            iterations += 1

        scheduler.step()

        if dist.get_rank() == 0:
            checkpoint = {
                "epochs": epochs + 1,
                "optimizer": optimizer.state_dict(),
                "model": resnet_model.module.state_dict(),
                "name": type(resnet_model.module).__name__,
                "config": resnet_model.extra_repr(),
            }
            torch.save(checkpoint, args.exp_dir + "/model" + str(epochs) + ".ckpt")
