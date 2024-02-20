import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMForCTC, AutoFeatureExtractor, Wav2Vec2Processor, HubertForCTC, HubertModel, AutoProcessor


class CustomHuBERTModel(nn.Module):
    # base: facebook/hubert-base-ls960
    # large: facebook/hubert-xlarge-ls960-ft, facebook/hubert-large-ls960-ft
    # large: facebook/hubert-xlarge-ll60k, facebook/hubert-large-ll60k

    def __init__(self, model_name="facebook/hubert-base-ls960"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)

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
