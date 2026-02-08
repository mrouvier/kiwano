import functools
import math
from collections import OrderedDict
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MaxPoolNd = {1: nn.MaxPool1d, 2: nn.MaxPool2d}

ConvNd = {1: nn.Conv1d, 2: nn.Conv2d}

BatchNormNd = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d}


class fwSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block
    link: https://arxiv.org/pdf/1709.01507.pdf
    PyTorch implementation
    """

    def __init__(self, num_freq, num_feats=64):
        super(fwSEBlock, self).__init__()
        self.squeeze = nn.Linear(num_freq, num_feats)
        self.exitation = nn.Linear(num_feats, num_freq)

        self.activation = nn.ReLU()  # Assuming ReLU, modify as needed

    def forward(self, inputs):
        # [bs, C, F, T]
        x = torch.mean(inputs, dim=[1, 3])
        x = self.squeeze(x)
        x = self.activation(x)
        x = self.exitation(x)
        x = torch.sigmoid(x)
        # Reshape and apply excitation
        x = x[:, None, :, None]
        x = inputs * x
        return x


class AMSMLoss(nn.Module):
    """
    Additive Margin Softmax (AM-Softmax) head.

    Computes cosine similarities between L2-normalized embeddings and class
    weights; subtracts margin m from the target-class logit; scales by s.

    Parameters
    ----------
    num_features : int
        Embedding dimensionality.
    num_classes : int
        Number of training speaker identities.
    s : float, default=30.0
        Logit scale.
    m : float, default=0.4
        Additive margin.

    Example
    -------
    >>> head = AMSMLoss(256, 6000, s=30.0, m=0.3)
    >>> x = torch.randn(16, 256)
    >>> # Inference: get cosine logits (e.g., for calibration/diagnostics)
    >>> logits = head(x)              # no labels
    >>> # Training: margin + scale applied
    >>> y = torch.randint(0, 6000, (16,))
    >>> logits = head(x, y)
    >>> loss = torch.nn.CrossEntropyLoss()(logits, y)
    """

    def __init__(self, num_features, num_classes, s=30.0, m=0.4):
        super(AMSMLoss, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def get_m(self):
        return self.m

    def get_s(self):
        return self.s

    def set_m(self, m):
        self.m = m

    def set_s(self, s):
        self.s = s

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def get_w(self):
        return self.W

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)

        # normalize weights
        W = F.normalize(self.W)

        # dot product
        logits = F.linear(x, W)
        if label == None:
            return logits

        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot

        # feature re-scale
        output *= self.s
        return output


class SpeakerEmbedding(nn.Module):
    """
    SpeakerEmbedding: Projection head for speaker embedding extraction.

    This module projects the pooled representation from the convolutional or
    temporal encoder (e.g., ASTP output) into a fixed-dimensional speaker
    embedding space. It performs feature normalization before and after the
    linear projection to stabilize training and improve embedding consistency.

    The structure is:
        BatchNorm1d (input) → Linear → BatchNorm1d (embedding)

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input feature vector (e.g., output of pooling layer).
    embed_dim : int
        Target dimensionality of the speaker embedding.

    Example
    -------
    >>> import torch
    >>> from model import SpeakerEmbedding
    >>>
    >>> # Example: pooled features from ASTP
    >>> pooled = torch.randn(32, 2816)  # batch=32, in_dim=2816
    >>>
    >>> # Create embedding head
    >>> embed_head = SpeakerEmbedding(in_dim=2816, embed_dim=256)
    >>>
    >>> # Forward pass
    >>> emb = embed_head(pooled)
    >>> emb.shape
    torch.Size([32, 256])
    >>>
    >>> # Normalize embeddings for cosine scoring
    >>> emb = torch.nn.functional.normalize(emb)
    """

    def __init__(self, in_dim, embed_dim):
        super(SpeakerEmbedding, self).__init__()

        self.in_dim = in_dim
        self.embed_dim = embed_dim

        self.norm_stats = torch.nn.BatchNorm1d(self.in_dim)
        self.fc_embed = nn.Linear(self.in_dim, self.embed_dim)
        self.norm_embed = torch.nn.BatchNorm1d(self.embed_dim)

    def forward(self, x):
        x = self.norm_stats(x)
        x = self.fc_embed(x)
        x = self.norm_embed(x)
        return x


class ASTP(nn.Module):
    """
    ASTP: Attentive Statistics Pooling for speaker embedding extraction.

    This module implements the *Attentive Statistics Pooling* mechanism. It
    aggregates variable-length frame-level representations into a fixed-dimensional
    utterance-level vector by computing attention-weighted mean and standard
    deviation statistics across the time axis.

    The attention mechanism learns a set of frame-level importance weights that
    highlight speaker-discriminative frames and suppress non-informative ones.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the frame-level feature input (number of channels).
    bottleneck_dim : int
        Dimensionality of the hidden attention layer (bottleneck).
        Typically a small fraction of `in_dim`.

    Example
    -------
    >>> import torch
    >>> from model import ASTP
    >>>
    >>> # Example input: feature map from a convolutional encoder
    >>> x = torch.randn(16, 512, 150)  # batch=16, channels=512, time=150
    >>>
    >>> # Create attentive pooling layer
    >>> astp = ASTP(in_dim=512, bottleneck_dim=256)
    >>>
    >>> # Forward pass
    >>> pooled = astp(x)
    >>> pooled.shape
    >>>
    >>> #torch.Size([16, 1024])  # Concatenation of mean (512) + std (512)
    """

    def __init__(self, in_dim, bottleneck_dim):
        super(ASTP, self).__init__()

        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim

        self.attention = nn.Sequential(
            nn.Conv1d(self.in_dim, self.bottleneck_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(self.bottleneck_dim, self.in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        w = self.attention(x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))

        return torch.cat([mu, sg], dim=1)


# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/wav2vec2/modeling_wav2vec2.py
# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Wav2Vec2
class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output


# -------------------------------------------------------------
# Import activations from hugginface
# -------------------------------------------------------------
# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py


def gelu(x):
    """This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class NewGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = F.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class ClippedGELUActivation(nn.Module):
    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py
class FastGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input))
            )
        )


# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py
class QuickGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py
class SiLUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input)


"""
class MishActivation(nn.Module):
    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = F.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(F.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)
"""


class LinearActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


# "mish": MishActivation,

ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "linear": LinearActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "swish": SiLUActivation,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)

# -------------------------------------------------------------
#  Copy FeedForward & Encoder from hugginface wav2vec2
# -------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu_new",
        activation_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[hidden_act]
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_mlp: int,
        n_head: int,
        channel_last: bool = False,
        act: str = "gelu_new",
        act_do: float = 0.0,
        att_do: float = 0.0,
        hid_do: float = 0.0,
        ln_eps: float = 1e-6,
    ):

        hidden_size = n_state
        num_attention_heads = n_head
        intermediate_size = n_mlp
        hidden_act = act
        activation_dropout = act_do
        attention_dropout = att_do
        hidden_dropout = hid_do
        layer_norm_eps = ln_eps

        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            activation_dropout=activation_dropout,
            hidden_dropout=hidden_dropout,
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        if not self.channel_last:
            hidden_states = hidden_states.permute(0, 2, 1)
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = hidden_states
        if not self.channel_last:
            outputs = outputs.permute(0, 2, 1)
        return outputs


# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
class ConvNeXtLikeBlock(nn.Module):
    def __init__(
        self,
        C,
        dim=2,
        kernel_sizes=[
            (3, 3),
        ],
        Gdiv=1,
        padding="same",
        activation="gelu",
    ):
        super().__init__()
        # if C//Gdiv==0:
        #     Gdiv = C
        self.dwconvs = nn.ModuleList(
            modules=[
                ConvNd[dim](
                    C,
                    C,
                    kernel_size=ks,
                    padding=padding,
                    groups=C // Gdiv if Gdiv is not None else 1,
                )
                for ks in kernel_sizes
            ]
        )
        self.norm = BatchNormNd[dim](C * len(kernel_sizes))
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        self.pwconv1 = ConvNd[dim](
            C * len(kernel_sizes), C, 1
        )  # pointwise/1x1 convs, implemented with linear layers

    def forward(self, x):
        skip = x
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs], dim=1)
        x = self.act(self.norm(x))
        x = self.pwconv1(x)
        x = skip + x
        return x


class to1d(nn.Module):
    def forward(self, x):
        size = x.size()
        # bs,c,f,t = tuple(size)
        bs, c, f, t = size
        return x.permute((0, 2, 1, 3)).reshape((bs, c * f, t))


class to2d(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self, x):
        size = x.size()
        # bs,cf,t = tuple(size)
        bs, cf, t = size
        out = x.reshape((bs, self.f, self.c, t)).permute((0, 2, 1, 3))
        # print(f"to2d : {out.size()}")
        return out

    def extra_repr(self) -> str:
        return f"f={self.f},c={self.c}"


class weigth1d(nn.Module):
    def __init__(self, N, C, sequential=False, requires_grad=True):
        super().__init__()
        self.N = N
        self.sequential = sequential
        self.w = nn.Parameter(torch.zeros(1, N, C, 1), requires_grad=requires_grad)

    def forward(self, xs):
        w = F.softmax(self.w, dim=1)
        if not self.sequential:
            xs = torch.cat([t.unsqueeze(1) for t in xs], dim=1)
            x = (w * xs).sum(dim=1)
            # print(f"weigth1d : {x.size()}")
        else:
            s = torch.zeros_like(xs[0])
            for i, t in enumerate(xs):
                s += t * w[:, i, :, :]
            x = s
            # x = sum([t*w[:,i,:,:] for i,t in enumerate(xs)])
        return x

    def extra_repr(self) -> str:
        return f"w={tuple(self.w.size())},sequential={self.sequential}"


class LayerNorm(nn.Module):  # ⚡
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, T, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, T).
    """

    def __init__(self, C, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.C = (C,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.C, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)

            w = self.weight
            b = self.bias
            for _ in range(x.ndim - 2):
                w = w.unsqueeze(-1)
                b = b.unsqueeze(-1)
            x = w * x + b  # ⚡
            return x

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"{k}={v}"
                for k, v in {
                    "C": self.C,
                    "data_format": self.data_format,
                    "eps": self.eps,
                }.items()
            ]
        )


class ResBasicBlock(nn.Module):
    def __init__(
        self, inc, outc, num_freq, stride=1, se_channels=64, Gdiv=4, use_fwSE=False
    ):
        super().__init__()
        # if inc//Gdiv==0:
        #     Gdiv = inc
        self.conv1 = nn.Conv2d(
            inc,
            inc if Gdiv is not None else outc,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=inc // Gdiv if Gdiv is not None else 1,
        )
        if Gdiv is not None:
            self.conv1pw = nn.Conv2d(inc, outc, 1)
        else:
            self.conv1pw = nn.Identity()

        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(
            outc,
            outc,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=outc // Gdiv if Gdiv is not None else 1,
        )

        if Gdiv is not None:
            self.conv2pw = nn.Conv2d(outc, outc, 1)
        else:
            self.conv2pw = nn.Identity()

        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)

        if use_fwSE:
            self.se = fwSEBlock(num_freq, se_channels)
        else:
            self.se = nn.Identity()

        if outc != inc:
            self.downsample = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outc),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1pw(self.conv1(x))
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2pw(self.conv2(out))
        out = self.bn2(out)
        out = self.se(out)

        out += self.downsample(residual)
        out = self.relu(out)
        # print(out.size())
        return out


class ConvBlock2d(nn.Module):
    def __init__(self, c, f, block_type="convnext_like", Gdiv=1):
        super().__init__()
        if block_type == "convnext_like":
            self.conv_block = ConvNeXtLikeBlock(
                c,
                dim=2,
                kernel_sizes=[(3, 3)],
                Gdiv=Gdiv,
                padding="same",
                activation="gelu",
            )
        elif block_type == "convnext_like_relu":
            self.conv_block = ConvNeXtLikeBlock(
                c,
                dim=2,
                kernel_sizes=[(3, 3)],
                Gdiv=Gdiv,
                padding="same",
                activation="relu",
            )
        elif block_type == "basic_resnet":
            self.conv_block = ResBasicBlock(
                c,
                c,
                f,
                stride=1,
                se_channels=min(64, max(c, 32)),
                Gdiv=Gdiv,
                use_fwSE=False,
            )
        elif block_type == "basic_resnet_fwse":
            self.conv_block = ResBasicBlock(
                c,
                c,
                f,
                stride=1,
                se_channels=min(64, max(c, 32)),
                Gdiv=Gdiv,
                use_fwSE=True,
            )
        else:
            raise NotImplemented()

    def forward(self, x):
        return self.conv_block(x)


# ------------------------------------------
#                1D block
# ------------------------------------------


class PosEncConv(nn.Module):
    def __init__(self, C, ks, groups=None):
        super().__init__()
        assert ks % 2 == 1
        self.conv = nn.Conv1d(
            C, C, ks, padding=ks // 2, groups=C if groups is None else groups
        )
        self.norm = LayerNorm(C, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        return x + self.norm(self.conv(x))


class TimeContextBlock1d(nn.Module):
    def __init__(
        self,
        C,
        hC,
        pos_ker_sz=59,
        block_type="att",
        red_dim_conv=None,
        exp_dim_conv=None,
    ):
        super().__init__()
        assert pos_ker_sz

        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(C, hC, 1), LayerNorm(hC, eps=1e-6, data_format="channels_first")
        )
        if block_type == "fc":
            self.tcm = nn.Sequential(
                nn.Conv1d(hC, hC * 2, 1),
                LayerNorm(hC * 2, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                nn.Conv1d(hC * 2, hC, 1),
            )
        elif block_type == "conv":
            # Just large kernel size conv like in convformer
            self.tcm = nn.Sequential(
                *[
                    ConvNeXtLikeBlock(
                        hC, dim=1, kernel_sizes=[7, 15, 31], Gdiv=1, padding="same"
                    )
                    for i in range(4)
                ]
            )
        elif block_type == "att":
            # Basic Transformer self-attention encoder block
            self.tcm = nn.Sequential(
                PosEncConv(hC, ks=pos_ker_sz, groups=hC),
                TransformerEncoderLayer(n_state=hC, n_mlp=hC * 2, n_head=4),
            )
        elif block_type == "conv+att":
            # Basic Transformer self-attention encoder block
            self.tcm = nn.Sequential(
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding="same"),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding="same"),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding="same"),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding="same"),
                TransformerEncoderLayer(n_state=hC, n_mlp=hC, n_head=4),
            )
        else:
            raise NotImplemented()

        self.exp_dim_conv = nn.Conv1d(hC, C, 1)

    def forward(self, x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x


class ReDimNetBackbone(nn.Module):
    def __init__(
        self,
        F=72,
        C=12,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            (1, 2, 1, [(3, 3)], None),
            (2, 3, 1, [(3, 3)], None),
            (3, 4, 1, [(3, 3)], 8),
            (2, 5, 1, [(3, 3)], 8),
            (1, 5, 1, [(7, 1)], 8),
            (2, 3, 1, [(3, 3)], 8),
        ],
        group_divisor=1,
        out_channels=512,
        feat_agg_dropout=0.0,
        offset_fm_weights=0,
    ):
        super().__init__()
        self.F = F
        self.C = C

        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type

        self.feat_agg_dropout = feat_agg_dropout
        self.offset_fm_weights = offset_fm_weights

        self.stages_setup = stages_setup
        self.build(F, C, stages_setup, group_divisor, out_channels, offset_fm_weights)

    def build(self, F, C, stages_setup, group_divisor, out_channels, offset_fm_weights):
        self.F = F
        self.C = C

        c = C
        f = F
        self.num_stages = len(stages_setup)

        self.stem = nn.Sequential(
            nn.Conv2d(1, int(c), kernel_size=3, stride=1, padding="same"),
            LayerNorm(int(c), eps=1e-6, data_format="channels_first"),
            to1d(),
        )

        Block1d = functools.partial(TimeContextBlock1d, block_type=self.block_1d_type)
        Block2d = functools.partial(ConvBlock2d, block_type=self.block_2d_type)

        self.stages_cfs = []
        for stage_ind, (
            stride,
            num_blocks,
            conv_exp,
            kernel_sizes,
            att_block_red,
        ) in enumerate(stages_setup):
            assert stride in [1, 2, 3]

            num_feats_to_weight = offset_fm_weights + stage_ind + 1
            layers = [
                weigth1d(
                    N=num_feats_to_weight,
                    C=F * C if num_feats_to_weight > 1 else 1,
                    requires_grad=num_feats_to_weight > 1,
                ),
                to2d(f=f, c=c),
                nn.Conv2d(
                    int(c),
                    int(stride * c * conv_exp),
                    kernel_size=(stride, 1),
                    stride=(stride, 1),
                    padding=0,
                    groups=1,
                ),
            ]

            self.stages_cfs.append((c, f))

            c = stride * c
            assert f % stride == 0
            f = f // stride

            for _ in range(num_blocks):
                layers.append(Block2d(c=int(c * conv_exp), f=f, Gdiv=group_divisor))

            if conv_exp != 1:
                _group_divisor = group_divisor
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            int(c * conv_exp),
                            c,
                            kernel_size=(3, 3),
                            stride=1,
                            padding="same",
                            groups=c // _group_divisor
                            if _group_divisor is not None
                            else 1,
                        ),
                        nn.BatchNorm2d(c, eps=1e-6),
                        nn.GELU(),
                        nn.Conv2d(c, c, 1),
                    )
                )

            layers.append(to1d())

            if att_block_red is not None:
                layers.append(Block1d(C * F, hC=(C * F) // att_block_red))

            setattr(self, f"stage{stage_ind}", nn.Sequential(*layers))

        num_feats_to_weight_fin = offset_fm_weights + len(stages_setup) + 1
        self.fin_wght1d = weigth1d(
            N=num_feats_to_weight_fin, C=F * C, requires_grad=num_feats_to_weight > 1
        )

        if out_channels is not None:
            self.mfa = nn.Sequential(
                nn.Conv1d(self.F * self.C, out_channels, kernel_size=1, padding="same"),
                nn.BatchNorm1d(out_channels, affine=True),
            )
        else:
            self.mfa = nn.Identity()

    def run_stage(self, prev_outs_1d, stage_ind):
        stage = getattr(self, f"stage{stage_ind}")
        return stage(prev_outs_1d)

    def forward(self, inp):
        x = self.stem(inp)
        outputs_1d = [x]
        for stage_ind in range(self.num_stages):
            outputs_1d.append(
                F.dropout(
                    self.run_stage(outputs_1d, stage_ind),
                    p=self.feat_agg_dropout,
                    training=self.training,
                )
            )

        x = self.fin_wght1d(outputs_1d)
        outputs_1d.append(x)
        x = self.mfa(x)
        return x


class ReDimNet(nn.Module):
    def __init__(
        self,
        F=72,
        C=12,
        block_1d_type="att",
        block_2d_type="convnext_like",
        stages_setup=[
            (1, 2, 1, [(3, 3)], None),
            (2, 3, 1, [(3, 3)], None),
            (3, 4, 1, [(3, 3)], 8),
            (2, 5, 1, [(3, 3)], 8),
            (1, 5, 1, [(7, 1)], 8),
            (2, 3, 1, [(3, 3)], 8),
        ],
        group_divisor=1,
        out_channels=512,
        feat_agg_dropout=0.0,
        offset_fm_weights=0,
        emb_dim=256,
        num_classes=6000,
    ):
        super().__init__()

        self.backbone = ReDimNetBackbone(
            F,
            C,
            block_1d_type,
            block_2d_type,
            stages_setup,
            group_divisor,
            out_channels,
            feat_agg_dropout,
            offset_fm_weights,
        )

        self.temporal_pooling = ASTP(out_channels, out_channels // 2)

        self.embedding = SpeakerEmbedding(out_channels * 2, emb_dim)

        self.output = AMSMLoss(emb_dim, num_classes)

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

    def forward(self, x, iden=None):
        x = self.backbone(x)
        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x
