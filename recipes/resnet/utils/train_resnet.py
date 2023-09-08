#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.models.resnet import conv3x3, conv1x1
import torch.nn.functional as F

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from typing import Union
import librosa

from torch.utils.data import DataLoader, Sampler

# add_reverb,add_noise,filtering,phone_filtering,codec

# add_reverb,add_noise,phone_filtering,codec


'''
train_dataset = K2SpeechRecognitionDataset(
    cut_transforms=[
        PerturbSpeed(factors=[0.9, 1.1], p=2 / 3),
        PerturbVolume(scale_low=0.125, scale_high=2.0, p=0.5),
        # [optional] you can supply noise examples to be mixed in the data
        CutMix(musan_cuts, snr=[10, 20], p=0.5),
        # [optional] you can supply RIR examples to reverberate the data
        ReverbWithImpulseResponse(rir_recordings, p=0.5),
    ],
    input_transforms=[
        SpecAugment(),  # default configuration is well-tuned
    ],
    input_strategy=OnTheFlyFeatures(Fbank()),
)
'''


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class AMSMLoss(nn.Module):

    def __init__(self, num_features, num_classes, s=30.0, m=0.4):
        super(AMSMLoss, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)

        # normalize weights
        W = F.normalize(self.W)

        # dot product
        logits = F.linear(x, W)
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot

        # feature re-scale
        output *= self.s
        return output


class ResNet(nn.Module):
    def __init__(self, embed_features=256, num_classes=6000):
        super(ResNet, self).__init__()

        self.embed_features = embed_features
        self.num_classes = num_classes

        self.pre_conv1 = nn.Conv2d(1, 128, 3, 1, 1, bias=False)
        self.pre_bn1 = nn.BatchNorm2d(128)
        self.pre_relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(128, 128, 3, stride=1)
        self.layer2 = self._make_layer(128, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 256, 3, stride=2)

        self.norm_stats = torch.nn.BatchNorm1d(2 * 8 * 256)

        self.fc_embed = nn.Linear(2 * 8 * 256, self.embed_features)
        self.norm_embed = torch.nn.BatchNorm1d(self.embed_features)

        self.attention = nn.Sequential(
            nn.Conv1d(256 * 8, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256 * 8, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.output = AMSMLoss(self.embed_features, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(BasicBlock(inchannel, outchannel, stride, downsample))

        for i in range(1, block_num):
            layers.append(BasicBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_conv1(x)
        x = self.pre_bn1(x)
        x = self.pre_relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(2, 3)
        x = x.flatten(1, 2)

        w = self.attention(x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
        x = torch.cat([mu, sg], dim=1)
        x = self.norm_stats(x)

        x = self.fc_embed(x)
        x = self.norm_embed(x)

        x = self.output(x)

        return x


class Augmentation():

    def __init__(self):
        pass


class Segment():
    segmentid: str
    duration: float
    spkid: str
    file_path: str

    def __init__(self, segmentid: str, spkid: str, duration: float, file_path: str):
        self.segmentid = segmentid
        self.spkid = spkid
        self.duration = duration
        self.file_path = file_path

    def compute_features(self):
        audio_data, sample_rate = librosa.load(self.file_path)
        fb = Fbank()
        audio_data = fb.extract(audio_data, sampling_rate=16000)
        audio_data = self._load_specific_size_from_extracted(audio_data, size)
        return audio_data

    def _load_specific_size_from_extracted(self, audio_data: np.ndarray, size: int):
        """
        Get specific duration within the extracted audio

        @param audio_data: the extracted audio data
        @type audio_data: np.ndarray
        @param size: the size of the frame to take
        @type size: int
        @return: the frame
        @rtype: np.ndarray
        """
        max_start_time = self.duration - size

        start_time = np.random.uniform(0.0, max_start_time)
        end_time = start_time + size

        start_sample = int(start_time * 100)
        end_sample = int(end_time * 100)

        return audio_data[start_sample:end_sample, :]


class SegmentSet():
    def __init__(self):
        self.segments = {}

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        print("ttt")
        if isinstance(segment_id_or_index, str):
            return self.segments[segment_id_or_index]
        return next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)

    def from_dict(self, target_dir: Pathlike):
        with open(target_dir / "liste") as f:
            for line in f:
                segmentid, spkid, duration, audio = line.strip().split(" ")
                self.segments[segmentid] = Segment(segmentid, spkid, (float)(duration), audio)

    def summarize(self):
        print(len(self.segments))


if __name__ == '__main__':
    s = SegmentSet()
    s.from_dict(Path("data/voxceleb1/"))
    f = s["id10001_J9lHsKG98U8_00007"].compute_features()
    print(f)
    print(f.shape)
