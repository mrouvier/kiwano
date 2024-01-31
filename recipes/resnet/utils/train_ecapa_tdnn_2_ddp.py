#!/usr/bin/env python3
import argparse
import glob
import os
import time
import warnings
from pathlib import Path
from typing import List, Union

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from kiwano.augmentation import Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop, Reverb, \
    SpecAugment, Augmentation, Permute
from kiwano.dataset import SegmentSet
from kiwano.features import Fbank
from kiwano.model import ECAPAModel2DDP
from kiwano.model.tools import init_args

warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')


class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    def __init__(self, audio_transforms: List[Augmentation] = None, feature_extractor=None,
                 feature_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, segment_id_or_index: Union[int, str]):
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)

        feature, sample_rate = segment.load_audio()
        if self.audio_transforms is not None:
            feature, sample_rate = self.audio_transforms(feature, sample_rate)

        if self.feature_extractor is not None:
            feature = self.feature_extractor.extract(feature, sampling_rate=sample_rate)

        if self.feature_transforms is not None:
            feature = self.feature_transforms(feature)

        return feature, self.labels[segment.spkid]


def init_eer(score_path):
    errs = []
    with open(score_path) as file:
        lines = file.readlines()
        for line in lines:
            parteer = line.split(',')[-1]
            parteer = parteer.split(' ')[-1]
            parteer = parteer.replace('%', '')
            parteer = float(parteer)
            if parteer not in errs:
                errs.append(parteer)
    return errs


def ddp_setup(rank: int, world_size: int, master_port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def print_info(rank, content):
    if rank == 0:
        print(content, flush=True)


def main_ddp(
        rank: int,
        world_size: int
):
    parser = argparse.ArgumentParser(description="ECAPA_trainer")
    # Training Settings
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Duration of the input segments, eg: 200 for 2 second')
    parser.add_argument('--max_epoch', type=int, default=80, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size')
    parser.add_argument('--n_cpu', type=int, default=20, help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')

    # Training and evaluation path/lists, save path
    parser.add_argument('--eval_list', type=str, default=f"db/voxceleb1/veri_test2.txt",
                        help='The path of the evaluation list: veri_test2.txt, list_test_all2.txt, list_test_hard2.txt'
                             'veri_test2.txt comes from '
                             'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
    parser.add_argument('--eval_path', type=str, default=f"db/voxceleb1/wav/",
                        help='The path of the evaluation data, eg:"data/voxceleb1/" in my case')
    parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path to save the score.txt and models')
    parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')

    # Model and Loss settings
    parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
    parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
    parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
    parser.add_argument('--n_class', type=int, default=5994, help='Number of speakers')
    parser.add_argument('--feat_dim', type=int, default=81, help='Dim of features')
    parser.add_argument('--master_port', type=str, default="54322", help='Master port')
    # Command
    parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')

    # Initialization
    args = parser.parse_args()
    args = init_args(args)
    args.gpu_id = rank
    # Define the data loader
    musan = SegmentSet()
    musan.from_dict(Path(f"data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    reverb = SegmentSet()
    reverb.from_dict(Path("data/rirs_noises/"))

    ddp_setup(rank, world_size, args.master_port)

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=Sometimes([
            Noise(musan_music, snr_range=[5, 15]),
            Noise(musan_speech, snr_range=[13, 20]),
            Noise(musan_noise, snr_range=[0, 15]),
            Codec(),
            Filtering(),
            Normal(),
            Reverb(reverb)
        ]),
        feature_extractor=Fbank(),
        feature_transforms=Linear([
            CMVN(),
            Crop(350),
            SpecAugment(),
            Permute()
        ]),
    )

    training_data.from_dict(Path(f"data/voxceleb2/"))
    training_sampler = DistributedSampler(training_data)
    trainLoader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=training_sampler,
        num_workers=args.n_cpu,
        drop_last=True
    )

    # Search for the exist models
    modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
    modelfiles.sort()

    # Only do evaluation, the initial_model is necessary
    if args.eval:
        s = ECAPAModel2DDP(**vars(args))
        print_info(rank, "Model %s loaded from previous state!" % args.initial_model)
        s.load_parameters(args.initial_model)
        EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, n_cpu=args.n_cpu)
        print_info(rank, "EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
        quit()

    # If initial_model is exist, system will train from the initial_model
    if args.initial_model != "":
        print_info(rank, "Model %s loaded from previous state!" % args.initial_model)
        s = ECAPAModel2DDP(**vars(args))
        s.load_parameters(args.initial_model)
        epoch = 1
        EERs = []

    # Otherwise, system will try to start from the saved model&epoch
    elif len(modelfiles) >= 1:
        print_info(rank, "Model %s loaded from previous state!" % modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = ECAPAModel2DDP(**vars(args))
        s.load_parameters(modelfiles[-1])
        EERs = init_eer(args.score_save_path)
    # Otherwise, system will train from scratch
    else:
        epoch = 1
        s = ECAPAModel2DDP(**vars(args))
        EERs = []

    if rank == 0:
        score_file = open(args.score_save_path, "a+")

    while True:
        # Training for one epoch
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader, sampler=training_sampler)

        # Evaluation every [test_step] epochs
        if rank == 0:
            if epoch % args.test_step == 0:
                s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch, delete=True)
                EERs.append(s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, n_cpu=args.n_cpu)[0])
                print(time.strftime("%Y-%m-%d %H:%M:%S"),
                      "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%" % (epoch, acc, EERs[-1], min(EERs)))
                score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n" % (
                    epoch, lr, loss, acc, EERs[-1], min(EERs)))
                score_file.flush()
                if EERs[-1] <= min(EERs):
                    s.save_parameters(args.model_save_path + "/best.model")

            if epoch >= args.max_epoch:
                destroy_process_group()  # clean up
                quit()

        epoch += 1


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        main_ddp,
        args=(world_size,),
        nprocs=world_size,  # Total number of process = number of gpus
    )
