import pdb
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from kiwano.dataset import SegmentSet
from kiwano.model import ECAPAModel
from kiwano.model.wav2vec2 import CustomWav2Vec2Model
from recipes.resnet.utils.train_resnet import SpeakerTrainingSegmentSet
import pdb
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop, \
    SpecAugment, Reverb
import torch
import sys


class Wav2Vec2Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    print("START Loading data")
    sys.stdout.flush()
    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    model_name = "facebook/wav2vec2-base-960h"
    model_wav2vec2 = CustomWav2Vec2Model(model_name)
    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=Sometimes([
            Noise(musan_music, snr_range=[5, 15]),
            Noise(musan_speech, snr_range=[13, 20]),
            Noise(musan_noise, snr_range=[0, 15]),
            Codec(),
            Filtering(),
            Normal()
        ]),
        feature_extractor=model_wav2vec2
    )
    training_data.from_dict(Path("data/voxceleb1/"))
    print("END Loading data")
    sys.stdout.flush()

    wav2vec2_outputs = []

    train_dataloader = DataLoader(training_data, batch_size=48, drop_last=True, shuffle=True, num_workers=10)
    iterator = iter(train_dataloader)
    # The wav2vec2 output
    print(f"START Wav2vec2 ")
    sys.stdout.flush()
    for feats, iden in train_dataloader:
        with torch.cuda.amp.autocast(enabled=True):
            preds = model_wav2vec2(feats)

    print(f"END Wav2vec2")
    sys.stdout.flush()
    wav2vec2_dataset = Wav2Vec2Dataset(wav2vec2_outputs)
    train_dataloader = DataLoader(wav2vec2_dataset, batch_size=128, drop_last=True, shuffle=True, num_workers=10)

    num_iterations = 5

    ecapa_tdnn_model = ECAPAModel(
        lr=0.001,
        lr_decay=0.97,
        channel_in=768,
        channel_size=1024,
        n_class=6000,
        loss_margin=0.2,
        loss_scale=30,
        test_step=1
    )
    print(f"START ECAPA-TDNN {nb_segments} iterations")
    sys.stdout.flush()
    for epoch in range(1, num_iterations + 1):
        print(f"\t [{epoch} / {num_iterations}]")
        sys.stdout.flush()
        loss, lr, acc = ecapa_tdnn_model.train_network(epoch=epoch, loader=train_dataloader)

    print(f"END ECAPA-TDNN")
    sys.stdout.flush()