import os
import pdb
import sys
import time

import numpy
import soundfile
import torch
import tqdm
from torch import nn

import torch.nn.functional as F

from kiwano.model.ecapa_tdnn import ECAPA_TDNN
from kiwano.model.loss import AAMsoftmax
from kiwano.model.tools import tuneThresholdfromScore, ComputeMinDcf, ComputeErrorRates


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
            pdb.set_trace()
            self.zero_grad()
            data = data.permute(0, 2, 1)  # We change dim [128, 300, 81] to [128, 81, 300]
            labels = torch.LongTensor(labels)
            print(f"Data: {data}")
            print(f"Labels: {labels}")
            speaker_embedding = self.speaker_encoder.forward(data)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            print(time.strftime("%m-%d %H:%M:%S") + \
                  " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                  " Loss: %.5f, ACC: %2.2f%%" % (loss / num, top1 / index * len(labels)))
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(eval_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0))

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float)
            data_2 = torch.FloatTensor(feats)
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.speaker_encoder.forward(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.speaker_encoder.forward(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
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