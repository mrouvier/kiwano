from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


def get_output_rep(hidden_states):
    nb_layers = len(hidden_states)
    sum_hiddens = torch.zeros_like(hidden_states[0])
    for layer in range(nb_layers):
        sum_hiddens += hidden_states[layer]

    sum_hiddens = (1 / nb_layers) * sum_hiddens
    return sum_hiddens


class CustomWav2Vec2Model(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def forward(self, x):
        x = self.processor(x, return_tensor="pt", sampling_rate=16_000)
        x = torch.tensor(x.input_values)
        with torch.no_grad():
            output = self.model(x)

        hidden_states = list(output.hidden_states)
        # hidden_states = [h.squeeze(dim=0) for h in hidden_states]

        output = get_output_rep(hidden_states)

        return output

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:

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
