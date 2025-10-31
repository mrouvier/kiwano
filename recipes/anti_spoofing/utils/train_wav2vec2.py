import sys
from pathlib import Path

import torch
from recipes.resnet.utils.train_resnet import SpeakerTrainingSegmentSet
from torch.utils.data import DataLoader
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)

from kiwano.augmentation import (
    CMVN,
    Codec,
    Compose,
    Crop,
    Filtering,
    Noise,
    Normal,
    OneOf,
)
from kiwano.dataset import SegmentSet
from kiwano.features import Fbank
from kiwano.model import ECAPAModel
from kiwano.model.wav2vec2 import CustomWav2Vec2Model

if __name__ == "__main__":
    musan = SegmentSet()
    musan.from_dict(Path("data/musan/"))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    model_name = "facebook/wav2vec2-base-960h"
    model = CustomWav2Vec2Model(model_name)
    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=OneOf(
            [
                Noise(musan_music, snr_range=[5, 15]),
                Noise(musan_speech, snr_range=[13, 20]),
                Noise(musan_noise, snr_range=[0, 15]),
                Codec(),
                Filtering(),
                Normal(),
            ]
        ),
        feature_extractor=model,
        # feature_extractor=Fbank(),
        feature_transforms=Compose(
            [
                CMVN(),
                # Crop(300)
            ]
        ),
    )

    training_data.from_dict(Path("data/voxceleb1/"))

    # train_dataloader = DataLoader(training_data, batch_size=128, drop_last=True, shuffle=True, num_workers=10)
    # iterator = iter(train_dataloader)

    iterator = iter(training_data)
    # load pretrained model
    num_iterations = 3

    print(num_iterations)
    for iterations in range(0, num_iterations):
        feats, iden = next(iterator)
        feats = feats.squeeze(dim=0)
        output = model(feats)
        print(output)
