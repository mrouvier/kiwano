#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
import time
from torch import nn

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop, SpecAugment, Reverb
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNet, IDRDScheduler

import soundfile as sf

from torch.utils.data import Dataset, DataLoader, Sampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler

import argparse

import idr_torch
import logging
import os


logger = logging.getLogger(__name__)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
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

        return feature, self.labels[ segment.spkid ]



if __name__ == '__main__':

    #os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--nproc_per_node", type=int)

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda")

    torch.distributed.init_process_group(backend='nccl', init_method='env://')


    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")


    reverb = SegmentSet()
    reverb.from_dict(Path("data/rirs_noises/"))


    training_data = SpeakerTrainingSegmentSet(
                                    audio_transforms=Sometimes( [
                                        Noise(musan_music, snr_range=[5,15]),
                                        Noise(musan_speech, snr_range=[13,20]),
                                        Noise(musan_noise, snr_range=[0,15]),
                                        Codec(),
                                        Filtering(),
                                        Normal(),
                                        Reverb(reverb)
                                    ] ),
                                    feature_extractor=Fbank(),
                                    feature_transforms=Linear( [
                                        CMVN(),
                                        Crop(350),
                                        SpecAugment(),
                                    ] ),
                                )


    training_data.from_dict(Path("data/voxceleb2/"))

    train_sampler = DistributedSampler(training_data, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)

    train_dataloader = DataLoader(training_data, batch_size=32, drop_last=True, shuffle=False, num_workers=30, sampler=train_sampler, pin_memory=True)
    iterator = iter(train_dataloader)


    resnet_model = ResNet()
    resnet_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)
    resnet_model.to(device)

    resnet_model = torch.nn.parallel.DistributedDataParallel(resnet_model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = torch.optim.SGD(resnet_model.parameters(), lr=0.2, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    scheduler = IDRDScheduler(optimizer, num_epochs=100, initial_lr=0.2, warm_up_epoch=5, plateau_epoch=15, patience=5)

    scaler = torch.cuda.amp.GradScaler(enabled=True)


    for epochs in range(0, 150):
        iterations = 0
        train_sampler.set_epoch(epochs)
        torch.distributed.barrier()
        for feats, iden in train_dataloader:

            feats = feats.unsqueeze(1)

            feats = feats.float().to(device)
            iden = iden.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                preds = resnet_model(feats, iden)
                loss = criterion(preds, iden)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #loss.backward()
            #optimizer.step()

            if iterations%100 == 0:
                msg = "{}: Epoch: [{}/{}] {}/{} \t C-Loss:{:.4f} \t LR : {:.8f} \t Margin : {:.4f}".format(time.ctime(), epochs, 150, iterations, len(train_dataloader), loss.item(), get_lr(optimizer), resnet_model.module.get_m())
                print(msg)

            iterations += 1

        scheduler.step()
        #resnet_model.module.set_m( scheduler.get_amsmloss() )
        if dist.get_rank() == 0:
            state = {
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'model': resnet_model.module.state_dict(),
            }
            torch.save(resnet_model.module.state_dict(), "exp/resnet/model"+str(epochs)+".mat")

