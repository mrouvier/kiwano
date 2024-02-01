#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import DataLoader, Dataset

from kiwano.augmentation import Noise, Codec, Filtering, Normal, Sometimes, Linear, CropWaveForm, Reverb, \
    Augmentation
from kiwano.dataset import SegmentSet
from kiwano.model import ECAPAModel
from kiwano.model.tools import init_args


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


if __name__ == '__main__':
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

    # Paths
    parser.add_argument('--eval_list', type=str, default=f"db/voxceleb1/veri_test2.txt",
                        help='The path of the evaluation list: veri_test2.txt, list_test_all2.txt, list_test_hard2.txt'
                             'veri_test2.txt comes from '
                             'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
    parser.add_argument('--eval_path', type=str, default=f"db/voxceleb1/wav/",
                        help='The path of the evaluation data, eg:"data/voxceleb1/" in my case')
    parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path to save the score.txt and models')
    parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')

    parser.add_argument('--musan_list_path', type=str, default="data/musan/",
                        help='Path where your musan list is')
    parser.add_argument('--rirs_noise_list_path', type=str, default="data/rirs_noises/",
                        help='Path where your rirs noise list is')
    parser.add_argument('--training_list_path', type=str, default="data/voxceleb2/",
                        help='Path where your training list is')

    # Model and Loss settings
    parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
    parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
    parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
    parser.add_argument('--n_class', type=int, default=5994, help='Number of speakers')
    parser.add_argument('--feat_type', type=str, default='fbank', help='Type of features: fbank, wav2vec2')
    parser.add_argument('--feat_dim', type=int, default=80, help='Dim of features: fbank(80), wav2vec2(768)')
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base-960h',
                        help='facebook/wav2vec2-base-960h, facebook/wav2vec2-large-960h'
                             'facebook/wav2vec2-large-robust-ft-libri-960h, facebook/wav2vec2-large-960h-lv60-self')
    parser.add_argument('--is_2d', dest='is_2d', action='store_true', help='2d learneable weight')

    # model_name
    # Command
    parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')

    # Initialization
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    args = init_args(args)

    # Define the data loader
    musan = SegmentSet()
    musan.from_dict(Path(args.musan_list_path))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    reverb = SegmentSet()
    reverb.from_dict(Path(args.rirs_noise_list_path))

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
        feature_transforms=Linear([
            CropWaveForm()
        ]),
    )

    training_data.from_dict(Path(args.training_list_path))
    trainLoader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu,
                             drop_last=True, pin_memory=True)

    # Search for the exist models
    modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
    modelfiles.sort()

    # Only do evaluation, the initial_model is necessary
    if args.eval:
        s = ECAPAModel(**vars(args))
        print("Model %s loaded from previous state!" % args.initial_model)
        sys.stdout.flush()
        s.load_parameters(args.initial_model)
        EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, n_cpu=args.n_cpu)
        print("EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
        sys.stdout.flush()
        quit()

    # If initial_model is exist, system will train from the initial_model
    if args.initial_model != "":
        print("Model %s loaded from previous state!" % args.initial_model)
        sys.stdout.flush()
        s = ECAPAModel(**vars(args))
        s.load_parameters(args.initial_model)
        epoch = 1
        EERs = []

    # Otherwise, system will try to start from the saved model&epoch
    elif len(modelfiles) >= 1:
        print("Model %s loaded from previous state!" % modelfiles[-1])
        sys.stdout.flush()
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = ECAPAModel(**vars(args))
        s.load_parameters(modelfiles[-1])
        EERs = init_eer(args.score_save_path)
    # Otherwise, system will train from scratch
    else:
        epoch = 1
        s = ECAPAModel(**vars(args))
        EERs = []

    score_file = open(args.score_save_path, "a+")

    while True:
        # Training for one epoch
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)

        # Evaluation every [test_step] epochs
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
            quit()

        epoch += 1
