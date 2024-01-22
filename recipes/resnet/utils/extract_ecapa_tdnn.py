import argparse
import glob
import sys
import warnings
from pathlib import Path
from typing import Union, List

import torch
from torch.utils.data import Dataset, DataLoader

from kiwano.augmentation import Augmentation, Linear, CMVN, CropWaveForm
from kiwano.dataset import Segment, SegmentSet
from kiwano.embedding import write_pkl
from kiwano.model import ECAPAModel
from kiwano.model.tools import init_args


class SpeakerExtractingSegmentSet(Dataset, SegmentSet):
    def __init__(self, audio_transforms: List[Augmentation] = None, feature_extractor=None,
                 feature_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]):
        segment, feature = None, None

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

        return feature, segment.segmentid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECAPA_extractor")
    # Training Settings
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Duration of the input segments, eg: 200 for 2 second')
    parser.add_argument('--max_epoch', type=int, default=80, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size')
    parser.add_argument('--n_cpu', type=int, default=20, help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')

    # Evaluation path/lists, save path
    parser.add_argument('--eval_list', type=str, default=f"db/voxceleb1/veri_test2.txt",
                        help='The path of the evaluation list: veri_test2.txt, list_test_all2.txt, list_test_hard2.txt'
                             'veri_test2.txt comes from '
                             'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
    parser.add_argument('--eval_path', type=str, default=f"db/voxceleb1/wav/",
                        help='The path of the evaluation data, eg:"data/voxceleb1/" in my case')
    parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path to save the score.txt and models')
    parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')
    parser.add_argument('--output_dir', type=str, default="exps/exp1/xvectors/xvectors_voxceleb2.pkl",
                        help='Where to save xvectors')
    # Model and Loss settings
    parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
    parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
    parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
    parser.add_argument('--n_class', type=int, default=5994, help='Number of speakers')
    parser.add_argument('--feat_type', type=str, default='fbank', help='Type of features: fbank, wav2vec2')
    parser.add_argument('--feat_dim', type=int, default=80, help='Dim of features: fbank(80), wav2vec2(768)')

    # Command
    parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')

    # Initialization
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    args = init_args(args)

    extracting_data = SpeakerExtractingSegmentSet(
        feature_transforms=Linear([
            CMVN(),
            CropWaveForm(),
        ]),
    )

    extracting_data.from_dict(Path(f"data/voxceleb2/"))
    trainLoader = DataLoader(extracting_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu,
                             drop_last=True)

    # Search for the exist models
    modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
    modelfiles.sort()

    # Load model
    s = ECAPAModel(**vars(args))
    print("Model %s loaded from previous state!" % args.initial_model)
    sys.stdout.flush()
    s.load_parameters(args.initial_model)
    xvectors = s.extract_xvectors(trainLoader)
    write_pkl(args.output_dir, xvectors)
