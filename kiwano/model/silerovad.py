from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class SileroVadModel(torch.nn.Module):
    def __init__(self):
        super(SileroVadModel, self).__init__()
        forward_basis_buffer = torch.load("forward_basis_buffer.pt")

        # hop_length = 128
        # filter_length = 256

        hop_length = 64
        filter_length = 128

        self.stft = SileroVadSTFT(forward_basis_buffer, hop_length, filter_length)
        self.encoder = SileroVadEncoder()
        self.decoder = SileroVadDecoder()
        self.decoder.eval()

        self.reset_states()

        self.sample_rates = [8000, 16000]

    def reset_states(self, batch_size=1):
        self._state = torch.zeros((2, batch_size, 128)).float()
        self._context = torch.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

    def _validate_input(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")
        """
        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:,::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")
        """
        return x, sr

    def call(self, x, sr):
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros(batch_size, context_size)

        x = torch.cat([self._context, x], dim=1)
        if sr in [8000, 16000]:
            out = self.stft(x)
            out = self.encoder(out)
            out, state = self.decoder(out, self._state)
            self._state = state
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        # out = torch.from_numpy(out)
        return out

    def audio_forward(self, x, sr: int):
        outs = []

        x, sr = self._validate_input(x, sr)
        self.reset_states()
        num_samples = 512 if sr == 16000 else 256

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), "constant", value=0.0)

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i : i + num_samples]
            out = self.call(wavs_batch, sr)
            outs.append(out.item())

        return outs

    def forward(self, x, state):
        out = self.stft(x)
        out = self.encoder(out)
        out, s = self.decoder(out, state)
        return out, s


class SileroVadBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SileroVadBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # self.se = nn.Identity()
        self.reparam_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1,
            stride=self.stride,
        )
        self.batch_norm = nn.BatchNorm1d(self.out_channels)
        self.activation = nn.ReLU()

    def load(self, name):
        self.reparam_conv.load_state_dict(torch.load(name))

    def forward(self, x: Tensor):
        # x = self.se(x)
        x = self.activation(self.batch_norm(self.reparam_conv(x)))
        return x


class SileroVadEncoder(nn.Module):
    def __init__(self):
        super(SileroVadEncoder, self).__init__()
        """
        self.layers0 = SileroVadBlock(129, 128, 3, 1)
        self.layers1 = SileroVadBlock(128, 64, 3, 2)
        self.layers2 = SileroVadBlock(64, 64, 3, 2)
        self.layers3 = SileroVadBlock(64, 128, 3, 1)
        """

        self.layers0 = SileroVadBlock(65, 128, 3, 1)
        self.layers1 = SileroVadBlock(128, 64, 3, 2)
        self.layers2 = SileroVadBlock(64, 64, 3, 2)
        self.layers3 = SileroVadBlock(64, 128, 3, 1)

        # self.layers0.load("conv0.pt")
        # self.layers1.load("conv1.pt")
        # self.layers2.load("conv2.pt")
        # self.layers3.load("conv3.pt")

    def forward(self, x):
        x = self.layers0(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        return x


class SileroVadDecoder(nn.Module):
    def __init__(self):
        super(SileroVadDecoder, self).__init__()

        self.rnn = nn.LSTMCell(128, 128)
        self.decoder = nn.Sequential(
            nn.Dropout(0.1), nn.ReLU(), nn.Conv1d(128, 1, kernel_size=1), nn.Sigmoid()
        )

        # self.rnn.load_state_dict( torch.load("rnn.pt") )
        # self.decoder.load_state_dict( torch.load("decoder.pt") )

    def forward(self, x, state=torch.zeros(0)):
        x = x.squeeze(-1)
        if len(state):
            h, c = self.rnn(x, (state[0], state[1]))
        else:
            h, c = self.rnn(x)

        x = h.unsqueeze(-1).float()
        state = torch.stack([h, c])
        x = self.decoder(x)
        return x, state


class SileroVadSTFT(torch.nn.Module):
    def __init__(self, forward_basis_buffer, hop_length, filter_length):
        super(SileroVadSTFT, self).__init__()
        self.forward_basis_buffer = forward_basis_buffer
        print(self.forward_basis_buffer.shape)

        self.hop_length = hop_length
        self.filter_length = filter_length

    def forward(self, input_data: Tensor) -> Tuple[Tensor, Tensor]:
        padded_input = torch.nn.functional.pad(input_data, [0, 64], mode="reflect")
        input_data0 = torch.unsqueeze(padded_input, 1)

        forward_transform = torch.conv1d(
            input_data0,
            self.forward_basis_buffer,
            bias=None,
            stride=[self.hop_length],
            padding=[0],
        )

        cutoff = int((self.filter_length / 2) + 1)

        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)

        phase = torch.atan2(imag_part, real_part)

        return magnitude  # , phase
