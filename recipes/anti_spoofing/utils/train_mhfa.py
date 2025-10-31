#!/usr/bin/env python3

import argparse
import logging
import math
import os
import random
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
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from kiwano.augmentation import (
    CMVN,
    Augmentation,
    Codec,
    Codec16kHzWithEffector,
    Compose,
    Crop,
    Filtering,
    Noise,
    Normal,
    OneOf,
    Reverb,
    SpecAugment,
    SpeedPerturbV3,
)
from kiwano.dataset import Segment, SegmentSet
from kiwano.features import Fbank
from kiwano.model import MHFALarge
from kiwano.utils import Pathlike

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):

    # Unzip the batch
    images, labels = zip(*batch)

    minimum = 200000
    for img in images:
        if img.shape[0] < minimum:
            minimum = img.shape[0]

    if minimum > 450:
        minimum = 450

    m = random.randint(250, minimum)

    c = Crop(m)
    s = SpecAugment()

    # Stack images to create a 4D tensor
    images = torch.stack([s(c(img)) for img in images])

    # Convert labels to a tensor
    labels = torch.tensor(labels)

    return images, labels


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    def __init__(self, audio_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms

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

        durate = 32240

        if audio.shape[0] < durate:
            n = math.ceil(self.duration / audio.shape[0])
            audio = audio.repeat(n, 1)

        max_start_time = audio.shape[0] - durate
        start_time = random.randint(0, max_start_time)
        result = audio[start_time : start_time + durate]

        return result, self.labels[segment.spkid]


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
    parser.add_argument("--plateau", type=int, default=3)
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
    musan_music = musan.get_speaker("music")

    reverb = SegmentSet()
    reverb.from_dict(Path(args.rirs_noises))

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=OneOf(
            [
                Noise(musan_noise, snr_range=[0, 15]),
                Noise(musan_music, snr_range=[5, 15]),
                Normal(),
                Codec16kHzWithEffector(),
                Reverb(reverb),
            ]
        )
    )

    training_data.from_dict(Path(args.training_corpus))
    training_data.truncate(min_duration=2.2, max_duration=200.0)
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

    # resnet_model = MHFA("cache/models--microsoft--wavlm-base-plus/snapshots/4c66d4806a428f2e922ccfa1a962776e232d487b/")
    resnet_model = MHFALarge("WavLM-Large.pt")
    # if args.checkpoint:
    #    resnet_model.load_state_dict(  checkpoint["model"]  )
    resnet_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)
    resnet_model.to(device)
    resnet_model.train(True)

    resnet_model = torch.nn.parallel.DistributedDataParallel(
        resnet_model, device_ids=[gpu_id], find_unused_parameters=True
    )  # , device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = torch.optim.AdamW(
        resnet_model.parameters(), lr=0.00002, weight_decay=0.1
    )

    if args.checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    criterion = torch.nn.CrossEntropyLoss()

    running_loss = [np.nan for _ in range(500)]

    for epochs in range(epochs_start, 100):
        iterations = 0
        train_sampler.set_epoch(epochs)
        torch.distributed.barrier()
        for feats, iden in train_dataloader:

            feats = feats.float().to(device)
            iden = iden.to(device)

            optimizer.zero_grad()

            preds = resnet_model(feats, iden)

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
                    100,
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

        if dist.get_rank() == 0:
            checkpoint = {
                "epochs": epochs + 1,
                "optimizer": optimizer.state_dict(),
                "model": resnet_model.module.state_dict(),
                "name": type(resnet_model.module).__name__,
                "config": resnet_model.extra_repr(),
            }
            torch.save(checkpoint, args.exp_dir + "/model" + str(epochs) + ".ckpt")
