#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import sys

import torch
from torch.utils.data import DataLoader

from kiwano.augmentation import Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop
from kiwano.dataset import SegmentSet
from kiwano.features import Fbank
from kiwano.model import ECAPAModel, ECAPAFeatureExtractor
from recipes.resnet.utils.train_resnet import SpeakerTrainingSegmentSet
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ECAPA_trainer")

    # Model settings
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Duration of the input segments, eg: 200 for 2 second')
    parser.add_argument('--max_epoch', type=int, default=80, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size')
    parser.add_argument('--n_cpu', type=int, default=50, help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')
    parser.add_argument("--channel_in", type=int, default=81, help='Input Channel')
    parser.add_argument("--channel_size", type=int, default=1024, help='Channel Size')
    parser.add_argument("--n_class", type=int, default=6000, help='Number of class')
    parser.add_argument("--loss_margin", type=float, default=0.2, help='Loss margin')
    parser.add_argument("--loss_scale", type=float, default=30, help='Loss Scale')

    # Data settings
    parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')
    parser.add_argument('--model_path', type=str, default="exps/ecapa/best.model",
                        help='Path to save the model')
    parser.add_argument('--score_path', type=str, default="exps/ecapa/score.txt",
                        help='Path to save the model')
    parser.add_argument('--eval_list', type=str, default="db/voxceleb1/voxceleb1_test_v2.txt",
                        help='The path of the evaluation list')
    parser.add_argument('--eval_path', type=str, default="db/voxceleb1/wav",
                        help='The path of the evaluation data')

    args = parser.parse_args()
    ecapa_tdnn_model = ECAPAModel(**vars(args))
    feature_extractor = ECAPAFeatureExtractor()

    if args.eval:
        print("START Evaluation on pretrained model")
        sys.stdout.flush()
        pretrain_score_file = open(args.score_path, "a+")
        ecapa_tdnn_model.load_parameters(args.model_path)
        eer, min_dcf = ecapa_tdnn_model.eval_network(eval_list=args.eval_list, eval_path=args.eval_path,
                                                     feature_extractor=feature_extractor, num_workers=args.n_cpu)
        pretrain_score_file.write("EER %2.2f%%, minDCF %.4f%%\n" % (eer, min_dcf))
        pretrain_score_file.flush()
        print("END")
        sys.stdout.flush()
        quit()

    print("START Loading data")
    sys.stdout.flush()
    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=Sometimes([
            Noise(musan_music, snr_range=[5, 15]),
            Noise(musan_speech, snr_range=[13, 20]),
            Noise(musan_noise, snr_range=[0, 15]),
            Codec(),
            Filtering(),
            Normal()
        ]),
        feature_extractor=feature_extractor
    )
    training_data.from_dict(Path("data/voxceleb1/"))

    # batch_size=400
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                  num_workers=30)

    num_iterations = 10
    print(f"START ECAPA-TDNN {num_iterations} iterations")
    sys.stdout.flush()

    score_file = open(args.score_path, "a+")
    best_eer = torch.inf
    for epoch in range(1, num_iterations + 1):
        print(f"\t [{epoch} / {num_iterations}]")
        sys.stdout.flush()
        loss, lr, acc = ecapa_tdnn_model.train_network(epoch=epoch, loader=train_dataloader)
        eer, _ = ecapa_tdnn_model.eval_network(eval_list=args.eval_list, eval_path=args.eval_path,
                                               feature_extractor=feature_extractor, num_workers=10)
        if eer < best_eer:
            best_eer = eer
            ecapa_tdnn_model.save_parameters(args.model_path)

        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              "%d epoch, ACC %2.2f%%, EER %2.2f%%, BestEER %2.2f%%" % (epoch, acc, eer, best_eer))
        sys.stdout.flush()

        score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, BestEER %2.2f%%\n" % (
            epoch, lr, loss, acc, eer, best_eer))
        score_file.flush()

    print(f"END ECAPA-TDNN")
    sys.stdout.flush()
