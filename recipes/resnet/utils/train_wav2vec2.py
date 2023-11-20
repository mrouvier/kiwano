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


class Wav2Vec2Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
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
    pdb.set_trace()
    training_data.from_dict(Path("data/voxceleb1/"))

    wav2vec2_outputs = []
    segments = training_data.segments
    nb_segments = len(segments)

    pdb.set_trace()
    # The wav2vec2 output
    for i, key in enumerate(training_data):
        with torch.no_grad():
            feats, iden = training_data[key]
            feats = feats.squeeze(dim=0)
            output = model_wav2vec2(feats)
            wav2vec2_outputs.append((output, iden))
        if i != 0 and i % 128 * 5 == 0:
            break
    pdb.set_trace()
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
    for epoch in range(1, num_iterations + 1):
        loss, lr, acc = ecapa_tdnn_model.train_network(epoch=epoch, loader=train_dataloader)
