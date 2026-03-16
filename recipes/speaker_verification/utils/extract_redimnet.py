#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from torch import nn
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
)
from kiwano.dataset import Segment, SegmentSet
from kiwano.embedding import SpeakerEmbeddingWriter, open_output_writer
from kiwano.features import Fbank, FbankConfig
from kiwano.model import ReDimNet
from kiwano.utils import Pathlike


class SpeakerExtractingSegmentSet(Dataset, SegmentSet):
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

        feature = feature.transpose(0, 1)

        return feature, segment.segmentid


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--rank",
        default=0,
        type=int,
    )

    parser.add_argument(
        "data_dir",
        type=str,
        help="data_dir",
    )

    parser.add_argument(
        "model",
        type=str,
        help="Model",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="pkl:output.pkl",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print("#" + " ".join(sys.argv[0:]))
    print("# Started at " + time.ctime())
    print("#")

    device = torch.device("cuda")

    fbcfg = FbankConfig(num_filters=71)

    extracting_data = SpeakerExtractingSegmentSet(
        feature_extractor=Fbank(fbcfg),
        feature_transforms=Compose(
            [
                CMVN(),
                Crop(1400, random=False),
            ]
        ),
    )

    extracting_data.from_dict(Path(args.data_dir))

    extracting_sampler = DistributedSampler(
        extracting_data, num_replicas=args.world_size, rank=args.rank
    )

    extracting_dataloader = DataLoader(
        extracting_data,
        batch_size=1,
        num_workers=10,
        sampler=extracting_sampler,
        pin_memory=True,
    )
    iterator = iter(extracting_dataloader)

    model = ReDimNet(
            F=72,
            C=32,
            block_1d_type="conv+att",
            block_2d_type="basic_resnet",
            stages_setup=(
                (1, 4, 4, [(3, 3)], 32),
                (2, 6, 2, [(3, 3)], 32),
                (1, 6, 2, [(3, 3)], 24),
                (3, 8, 1, [(3, 3)], 24),
                (1, 8, 1, [(3, 3)], 16),
                (2, 8, 1, [(3, 3)], 16),
                ),
            group_divisor=32,
            out_channels=256,
            num_classes=18000,
            )
    model.load_state_dict(torch.load(args.model)["model"])
    model.to(device)

    model.eval()

    emb = open_output_writer(args.output_dir)

    for feat, key in extracting_dataloader:
        feat = feat.unsqueeze(1)

        feat = feat.float().to(device)

        pred = model(feat)

        emb.write(key[0], torch.Tensor(pred.cpu().detach()[0]))

        print("Processed x-vector for key : " + key[0])

    print("# Ended at " + time.ctime())
