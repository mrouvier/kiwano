import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMForCTC, AutoProcessor, AutoFeatureExtractor


class CustomWavLMModel(nn.Module):
    # base: microsoft/wavlm-base, microsoft/wavlm-base-plus, microsoft/wavlm-base-plus-sv, microsoft/wavlm-base-sv
    # large: microsoft/wavlm-large
    def __init__(self, model_name="microsoft/wavlm-base"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WavLMForCTC.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)

    def forward(self, x, learnable_weights):
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
            result += hidden * weights

        return result

    def get_output_dim(self):
        x = [torch.randn(16_000)]
        with torch.no_grad():
            x = self.processor(x, return_tensor='pt', sampling_rate=16_000)
            x = x.input_values[0]
            x = torch.tensor(x).to(self.device)
            output = self.model(x)

        hidden_states = list(output.hidden_states)
        n_layers = len(hidden_states)
        feat_dim = hidden_states[0].shape[-1]

        return n_layers, feat_dim
