from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor


def get_output_rep(hidden_states, learnable_weigths, n_layers, n_frames):
    sum_hiddens = torch.zeros(size=(1, n_frames))
    for layer in range(n_layers):
        sum_hiddens += learnable_weigths[layer] @ hidden_states[layer]
    return sum_hiddens


class CustomWav2Vec2Model(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, output_hidden_states=True)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def forward(self, x):
        x = self.processor(x, return_tensor='pt', sampling_rate=16_000)
        x = x.input_values[0]
        x = torch.tensor(x)
        x = x.unsqueeze(0)
        with torch.no_grad():
            output = self.model(x)

        hidden_states = list(output.hidden_states)
        # hidden_states = [h.squeeze(dim=0) for h in hidden_states]
        state_dict = self.model.state_dict()
        input_size = hidden_states[0].shape[1]  # Number of frames
        n_layers = len(hidden_states)  # Number of layers
        n_frames = hidden_states[0].shape[-1]
        learnable_weights = [torch.randn(size=(input_size,)) for _ in range(n_layers)]
        output = get_output_rep(hidden_states, learnable_weights, n_layers, n_frames)

        return output

    def extract(self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> Union[np.ndarray, torch.Tensor]:

        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        if is_numpy:
            return samples.cpu().numpy()
        else:
            return samples
