import pdb
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor

from kiwano.augmentation import CropWaveForm
import torch.nn.functional as F


class CustomWav2Vec2Model(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def forward(self, x, learnable_weights, is_2d=False):
        with torch.no_grad():
            x = self.processor(x, return_tensor='pt', sampling_rate=16_000)
            x = x.input_values[0]
            x = torch.tensor(x).to(self.device)
            output = self.model(x)
            learnable_weights = F.softmax(learnable_weights, dim=-1)

        hidden_states = list(output.hidden_states)
        hidden0 = hidden_states[0].permute(0, 2, 1)
        result = torch.zeros_like(hidden0)
        for i, hidden in enumerate(hidden_states):
            hidden = hidden.permute(0, 2, 1)
            weights = learnable_weights[i]
            if is_2d:
                weights = weights.unsqueeze(0).unsqueeze(-1)
                weights = weights.to(self.device)

            result += weights * hidden

        return result

    def get_output_dim(self):
        x = torch.randn(1, 160000)
        with torch.no_grad():
            x = self.processor(x, return_tensor='pt', sampling_rate=16_000)
            x = x.input_values[0]
            x = torch.tensor(x).to(self.device)
            output = self.model(x)

        hidden_states = list(output.hidden_states)
        n_layers = len(hidden_states)
        feat_dim = hidden_states[0].shape[-1]

        print(f"n_layers: {n_layers}, feat_dim: {feat_dim}")
        sys.stdout.flush()
        return n_layers, feat_dim
