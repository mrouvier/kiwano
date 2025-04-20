#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import hostlist
import idr_torch
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from silero_vad import (
    collect_chunks,
    get_speech_timestamps,
    load_silero_vad,
    read_audio,
)
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
from kiwano.model import IDRDScheduler, JeffreysLoss, ResNetASVSpoof
from kiwano.utils import Pathlike

logger = logging.getLogger(__name__)

silero_model = load_silero_vad()


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

    hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
    os.environ["MASTER_ADDR"] = hostnames[0]
    os.environ["MASTER_PORT"] = "29500"
    rank = int(os.environ["SLURM_NODEID"])
    world = int(os.environ["SLURM_JOB_NUM_NODES"])
    master_addr = hostnames[0]
    port = int(os.environ["MASTER_PORT"])
    checkpoint = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--musan", type=str, default="data/musan/")
    parser.add_argument("--rirs_noises", type=str, default="data/rirs_noises/")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--step_size", type=float, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("training_corpus", type=str, metavar="training_corpus")
    parser.add_argument("exp_dir", type=str, metavar="exp_dir")

    args = parser.parse_args()

    print("#" + " ".join(sys.argv[0:]))
    print("# Started at " + time.ctime())
    print("#")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location={"cuda": "cpu"})

    epochs_start = 0
    if args.checkpoint:
        epochs_start = checkpoint["epochs"]

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=idr_torch.rank,
        world_size=idr_torch.size,
    )

    rank = dist.get_rank()
    gpu_id = rank % torch.cuda.device_count()

    print(torch.cuda.device_count())
    print(gpu_id)

    device = torch.device(gpu_id)

    print("rank : " + str(rank))
    print("gpu_id : " + str(gpu_id))

    musan = SegmentSet()
    musan.from_dict(Path(args.musan))

    musan_noise = musan.get_speaker("noise")

    reverb = SegmentSet()
    reverb.from_dict(Path(args.rirs_noises))

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=OneOf(
            [Noise(musan_noise, snr_range=[0, 15]), Normal(), Reverb(reverb)]
        ),
        feature_extractor=Fbank(),
        feature_transforms=Compose(
            [
                CMVN(),
                Crop(400),
                SpecAugment(),
            ]
        ),
    )

    training_data.from_dict(Path(args.training_corpus))
    training_data.truncate(min_duration=4.0, max_duration=200.0)
    training_data.describe()

    train_sampler = DistributedSampler(
        training_data,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=64,
        drop_last=True,
        shuffle=False,
        num_workers=8,
        sampler=train_sampler,
        pin_memory=False,
    )
    iterator = iter(train_dataloader)

    resnet_model = ResNetASVSpoof(num_classes=2)
    if args.checkpoint:
        resnet_model.load_state_dict(checkpoint["model"])
    resnet_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)
    resnet_model.to(device)

    resnet_model = torch.nn.parallel.DistributedDataParallel(
        resnet_model, device_ids=[gpu_id]
    )  # , device_ids=[args.local_rank], output_device=args.local_rank)

    # optimizer = torch.optim.Adam(resnet_model.parameters(), lr = 0.0001, weight_decay = 1e-3)
    optimizer = torch.optim.SGD(
        resnet_model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    criterion = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    """
    scheduler = WarmupLRScheduler(
        optimizer, args.init_lr, args.peak_lr, args.warmup_steps
    )
    """
    if args.checkpoint:
        scheduler.step(checkpoint["epochs"])

    running_loss = [np.nan for _ in range(500)]

    for epochs in range(epochs_start, 50):
        iterations = 0
        train_sampler.set_epoch(epochs)
        torch.distributed.barrier()
        for feats, iden in train_dataloader:

            feats = feats.unsqueeze(1)

            feats = feats.float().to(device)
            iden = iden.to(device)

            # print(iden)

            optimizer.zero_grad()

            preds = resnet_model(feats, iden)

            # print(torch.argmax(preds, dim=1))

            # print("-----")

            loss = criterion(preds, iden)

            loss.backward()
            optimizer.step()

            running_loss.pop(0)
            running_loss.append(loss.item())
            rmean_loss = np.nanmean(np.array(running_loss))

            if iterations % 100 == 0:
                matches = (torch.argmax(preds, dim=1) == iden).sum()
                acc = matches / len(iden)

                msg = "{}: Epoch: [{}/{}] ({}/{}) \t AvgLoss:{:.4f} \t C-Loss:{:.4f} \t LR : {:.8f} \t Margin : {:.4f} \t Accuracy : {:.4f}".format(
                    time.ctime(),
                    epochs,
                    50,
                    iterations,
                    len(train_dataloader),
                    rmean_loss,
                    loss.item(),
                    get_lr(optimizer),
                    resnet_model.module.get_m(),
                    acc,
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
