import glob
import os
import pdb
import random
import sys
import time

import numpy
import soundfile
import torch
import tqdm
from scipy import signal
from torch import nn

import torch.nn.functional as F

from kiwano.model.ecapa_tdnn import ECAPA_TDNN
from kiwano.model.loss import AAMsoftmax
from kiwano.model.tools import tuneThresholdfromScore, ComputeMinDcf, ComputeErrorRates
from torch.utils.data import Dataset, DataLoader


class ECAPAValidateDataset(Dataset):
    def __init__(self, file_list, eval_path, feature_extractor, speaker_encoder):
        self.file_list = file_list
        self.eval_path = eval_path
        self.feature_extractor = feature_extractor
        self.speaker_encoder = speaker_encoder

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio, _ = soundfile.read(os.path.join(self.eval_path, self.file_list[idx]))
        # Full utterance
        data_1 = numpy.stack([audio], axis=0)
        data_1 = torch.FloatTensor(data_1[0])
        if self.feature_extractor:
            data_1 = self.feature_extractor(data_1)
        data_1 = data_1.unsqueeze(dim=0)

        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
        for asf in startframe:
            audio_ = audio[int(asf):int(asf) + max_audio]
            audio_ = torch.FloatTensor(audio_)
            if self.feature_extractor:
                audio_ = self.feature_extractor(audio_)
            feats.append(audio_)
        feats = numpy.stack(feats, axis=0).astype(numpy.float32)
        data_2 = torch.FloatTensor(feats)
        # Speaker embeddings
        with torch.no_grad():
            embedding_1 = self.speaker_encoder.forward(data_1)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = self.speaker_encoder.forward(data_2)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        return self.file_list[idx], [embedding_1, embedding_2]


class ECAPATrainDataset(Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            key = file.split('/')[-3]
            if key not in self.noiselist:
                self.noiselist[key] = []
            self.noiselist[key].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[1] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            line_parts = line.split()
            speaker_id = line_parts[1]
            audio_path = line_parts[-1]
            speaker_label = dictkeys[speaker_id]
            self.data_label.append(speaker_label)
            self.data_list.append(audio_path)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        # Data Augmentation
        augtype = random.randint(0, 5)
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')

        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, channel_in, channel_size, n_class, loss_margin, loss_scale, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(in_channels=channel_in, out_channels=channel_size)
        ## Classifier
        self.speaker_loss = AAMsoftmax(n_class=n_class, loss_margin=loss_margin, loss_scale=loss_scale)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        # print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
        #         sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels)
            speaker_embedding = self.speaker_encoder.forward(data, aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            print(time.strftime("%m-%d %H:%M:%S") + \
                  " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                  " Loss: %.5f, ACC: %2.2f%%" % (loss / num, top1 / index * len(labels)))
            sys.stdout.flush()
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, eval_path, feature_extractor):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        eval_dataset = ECAPAValidateDataset(setfiles, eval_path, feature_extractor, self.speaker_encoder)
        eval_dataloader = DataLoader(eval_dataset, batch_size=100, drop_last=False, shuffle=False, num_workers=5)
        for idx, (keys, values) in tqdm.tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            embeddings_1 = values[0]
            embeddings_2 = values[1]
            for i, key in enumerate(keys):
                embeddings[key] = [embeddings_1[i], embeddings_2[i]]

        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
