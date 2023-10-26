#!/usr/bin/env python3

import sys, os
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
import time
from torch import nn

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, CropNotRandom, Crop
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNet
from kiwano.embedding import EmbeddingSet, write_pkl

from torch.utils.data.distributed import DistributedSampler

import soundfile as sf

from torch.utils.data import Dataset, DataLoader, Sampler

import argparse

class SpeakerExtractingSegmentSet(Dataset, SegmentSet):
    def __init__(self, audio_transforms: List[Augmentation] = None, feature_extractor = None, feature_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        segment = None
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)

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
        "--rank",
        default=1,
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



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print("#"+" ".join( sys.argv[0:]  ))

    device = torch.device("cuda")


    extracting_data = SpeakerExtractingSegmentSet(
                                    feature_extractor=Fbank(),
                                    feature_transforms=Linear( [
                                        CMVN(),
                                        #Crop(300),
                                        CropNotRandom(1400)
                                    ] ),
                                )

    extracting_data.from_dict(Path(args.data_dir))

    extracting_sampler = DistributedSampler(extracting_data, num_replicas=args.world_size, rank=args.rank)

    extracting_dataloader = DataLoader(extracting_data, batch_size=1, num_workers=10, sampler=extracting_sampler, pin_memory=True)
    iterator = iter(extracting_dataloader)

    resnet_model = ResNet()
    #resnet_model.load_state_dict(torch.load(args.model))
    resnet_model.load_state_dict(torch.load(args.model)["model"])
    resnet_model.to(device)

    resnet_model.eval()


    emb = EmbeddingSet()

    for feat, key in extracting_dataloader:
        feat = feat.unsqueeze(1)#.unsqueeze(1)

        feat = feat.float().to(device)

        pred = resnet_model(feat)

        emb[key[0]] = torch.Tensor( pred.cpu().detach()[0] )

        print(key[0]+" extrated -- ")

    write_pkl(args.output_dir, emb)




