#!/usr/bin/env python3

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
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
from kiwano.embedding import EmbeddingSet, write_pkl
from kiwano.features import Fbank
from kiwano.model import ResNet, ResNetV2, ResNetV3, ResNetV4
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

        return feature, segment.segmentid


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--chunk",
        default=10,
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

    extracting_data = SpeakerExtractingSegmentSet(
        feature_extractor=Fbank(),
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

    # resnet_model = ResNet(num_classes=18000)
    resnet_model = ResNetV2()
    # resnet_model = ResNetV3(k=3)
    # resnet_model = ResNetV4()
    resnet_model.load_state_dict(torch.load(args.model)["model"])
    resnet_model.to(device)

    resnet_model.eval()

    emb = EmbeddingSet()

    chunk = Compose([Crop(400), CMVN()])

    for feat, key in extracting_dataloader:

        for i in range(0, 10):
            new_feat = copy.copy(chunk(feat[0]))
            new_feat = new_feat.unsqueeze(0).unsqueeze(1)

            new_key = copy.copy([key[0] + "#" + str(i)])

            new_feat = new_feat.float().to(device)

            pred = resnet_model(new_feat)

            emb[new_key[0]] = torch.Tensor(pred.cpu().detach()[0])

            print("Processed x-vector for key : " + new_key[0])

    write_pkl(args.output_dir, emb)

    print("# Ended at " + time.ctime())
