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
import torch
import torch.distributed as dist
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
from kiwano.model import JeffreysLoss, KiwanoResNet, WarmupPlateauScheduler
from kiwano.utils import Pathlike

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_lr(optimizer: torch.optim.Optimizer) -> float:
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
        if self.audio_transforms is not None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)

        if self.feature_extractor is None:
            raise RuntimeError(
                "feature_extractor must be set for SpeakerTrainingSegmentSet"
            )

        feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)

        if self.feature_transforms is not None:
            feature = self.feature_transforms(feature)

        return feature, self.labels[segment.spkid]


def validate_directories(args) -> None:
    def check_dir(path: str, name: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{name} directory does not exist: {path}")
        if not p.is_dir():
            raise NotADirectoryError(
                f"{name} must be a directory, but is a file: {path}"
            )

    check_dir(args.musan, "MUSAN")
    check_dir(args.rirs_noises, "RIRS_NOISES")
    check_dir(args.training_corpus, "TRAINING_CORPUS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiwano ResNet Speaker Training")

    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--musan", type=str, default="data/musan/")
    parser.add_argument("--rirs_noises", type=str, default="data/rirs_noises/")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--max_epochs", type=int, default=51)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=15)
    parser.add_argument(
        "--stage_blocks",
        type=str,
        default="3,4,6,3",
        help='Block configuration, e.g. "4,30,30,4"',
    )
    parser.add_argument(
        "--stage_channels",
        type=str,
        default="32,64,128,256",
        help='Channel configuration, e.g. "32,64,128,256"',
    )
    parser.add_argument(
        "--stage_strides",
        type=str,
        default="1,2,2,2",
        help='Stride configuration, e.g. "1,2,2,2"',
    )

    parser.add_argument(
        "--num_classes", type=int, default=6000, help="Number of classes (integer)"
    )
    parser.add_argument("training_corpus", type=str, metavar="training_corpus")
    parser.add_argument("exp_dir", type=str, metavar="exp_dir")
    return parser.parse_args()


def load_checkpoint_if_any(
    checkpoint_path: Optional[str],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[WarmupPlateauScheduler] = None,
    map_location: str = "cpu",
) -> int:
    if not checkpoint_path:
        return 0

    if not os.path.isfile(checkpoint_path):
        logger.warning(
            "Checkpoint %s not found, training from scratch", checkpoint_path
        )
        return 0

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(map_location))

    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "epoch" in checkpoint:
        scheduler.set_epoch(checkpoint["epoch"])

    return int(checkpoint.get("epoch", 0))


def init_distributed() -> None:
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=idr_torch.rank,
        world_size=idr_torch.size,
    )
    torch.cuda.set_device(idr_torch.local_rank)


def main() -> None:

    args = parse_args()
    validate_directories(args)
    args.blocks = list(map(int, args.blocks.split(",")))
    args.feature_maps = list(map(int, args.feature_maps.split(",")))

    print("#" + " ".join(sys.argv[0:]))
    print("# Started at " + time.ctime())
    print("#")

    node_id = os.getenv("SLURM_NODEID", "N/A")
    master_addr = os.getenv("MASTER_ADDR", "N/A")

    setup_logging()

    if idr_torch.rank == 0:
        logger.info("MASTER_ADDR = %s", master_addr)
        logger.info("MASTER_PORT = %s", str(idr_torch.master_port))
        logger.info(
            "Training on %d nodes and %d processes, master node: %s",
            len(idr_torch.hostname),
            idr_torch.size,
            master_addr,
        )
    logger.info(
        "Process %d corresponds to GPU %d of node %s",
        idr_torch.rank,
        idr_torch.local_rank,
        node_id,
    )

    init_distributed()
    device = torch.device("cuda")

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
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    resnet_model = KiwanoResNet(
        num_classes=args.num_classes,
        stage_channels=args.stage_channels,
        stage_blocks=args.stage_blocks,
        stage_strides=args.stage_strides,
    )
    resnet_model.to(device)
    resnet_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)
    resnet_model = torch.nn.parallel.DistributedDataParallel(
        resnet_model,
        device_ids=[idr_torch.local_rank],
        output_device=idr_torch.local_rank,
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

    start_epoch = load_checkpoint_if_any(
        args.checkpoint,
        model=resnet_model.module,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location="cpu",
    )

    running_loss = torch.full((500,), float("nan"), device="cpu")

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(start_epoch, args.max_epochs):
        iterations = 0
        train_sampler.set_epoch(epoch)
        resnet_model.module.set_m(scheduler.get_margin_loss())

        torch.distributed.barrier()

        for feats, iden in train_dataloader:

            feats = feats.unsqueeze(1).float().to(device, non_blocking=True)
            iden = iden.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=True):
                preds = resnet_model(feats, iden)
                loss = criterion(preds, iden)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss = torch.roll(running_loss, shifts=-1)
            running_loss[-1] = loss.detach().cpu()

            rmean_loss = torch.nanmean(running_loss).item()

            if iterations % 100 == 0:
                logger.info(
                    "Epoch: [%d/%d] (%d/%d)\tAvgLoss: %.4f\tLoss: %.4f\tLR: %.8f\tMargin: %.4f",
                    epoch,
                    args.max_epochs,
                    iterations,
                    len(train_dataloader),
                    rmean_loss,
                    loss.item(),
                    get_lr(optimizer),
                    resnet_model.module.get_m(),
                )

            iterations += 1

        scheduler.step()

        if dist.get_rank() == 0:
            os.makedirs(args.exp_dir, exist_ok=True)
            checkpoint = {
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "model": resnet_model.module.state_dict(),
                "name": type(resnet_model.module).__name__,
                "config": resnet_model.module.extra_repr(),
            }
            ckpt_path = os.path.join(args.exp_dir, f"model{epoch}.ckpt")
            torch.save(checkpoint, ckpt_path)
            logger.info("Checkpoint saved to %s", ckpt_path)


if __name__ == "__main__":
    main()
