import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """
    3×3 convolution with padding.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    out_planes : int
        Number of output channels.
    stride : int, default=1
        Convolution stride.

    Example
    -------
    >>> layer = conv3x3(64, 128, stride=2)
    >>> x = torch.randn(8, 64, 100, 81)
    >>> y = layer(x)
    >>> y.shape
    torch.Size([8, 128, 50, 41])
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """
    1×1 convolution (channel projection).

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    out_planes : int
        Number of output channels.
    stride : int, default=1
        Convolution stride.

    Example
    -------
    >>> proj = conv1x1(256, 128)
    >>> x = torch.randn(4, 256, 64, 21)
    >>> proj(x).shape
    torch.Size([4, 128, 64, 21])
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention.

    Applies channel-wise reweighting: z = σ(W2 δ(W1 GAP(x))) and outputs x ⊙ z.

    Parameters
    ----------
    channel : int
        Number of channels in the input.
    reduction : int, default=1
        Reduction ratio for the bottleneck MLP (channel // reduction).

    Attributes
    ----------
    avg_pool : nn.AdaptiveAvgPool2d
        Global average pooling to 1×1.
    fc : nn.Sequential
        Two-layer MLP with SiLU and Sigmoid producing channel gates.

    Example
    -------
    >>> se = SELayer(channel=256, reduction=4)
    >>> x = torch.randn(2, 256, 100, 20)
    >>> y = se(x); y.shape
    torch.Size([2, 256, 100, 20])
    """

    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlockPosition(nn.Module):
    """
    Pre-activation residual block with SE attention (position variant).

    Ordering (per conv): BN → SiLU → Conv. Final 1×1 conv and SE attention
    are applied before the residual addition.

    Parameters
    ----------
    inplanes : int
        Input channels.
    planes : int
        Output channels for the block.
    reduction : int, default=1
        SE bottleneck reduction ratio.
    stride : int, default=1
        Stride of the first 3×3 convolution.
    downsample : nn.Module or None, default=None
        Optional projection for the residual path when shape/stride changes.

    Example
    -------
    >>> blk = SEBasicBlockPosition(128, 128, reduction=4, stride=1)
    >>> x = torch.randn(8, 128, 150, 81)
    >>> y = blk(x); y.shape
    torch.Size([8, 128, 150, 81])
    """

    def __init__(self, inplanes, planes, reduction=1, stride=1, downsample=None):
        super(SEBasicBlockPosition, self).__init__()

        self.activation = nn.ReLU()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(planes, reduction)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class BasicBlockPosition(nn.Module):
    """
    Pre-activation residual block (position variant, no SE).

    Same ordering as `SEBasicBlockPosition` (BN → SiLU → Conv) but without
    the SE attention module.

    Parameters
    ----------
    inplanes : int
        Input channels.
    planes : int
        Output channels.
    stride : int, default=1
        Stride for the first 3×3 conv.
    downsample : nn.Module or None, default=None
        Optional residual projection.

    Example
    -------
    >>> blk = BasicBlockPosition(256, 256)
    >>> x = torch.randn(4, 256, 120, 81)
    >>> blk(x).shape
    torch.Size([4, 256, 120, 81])
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockPosition, self).__init__()

        self.activation = nn.ReLU()

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
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class SEBasicBlock(nn.Module):
    """
    Pre-activation residual block with SE attention.

    Identical to `SEBasicBlockPosition` in computation; named variant for
    stacks that interleave SE and non-SE blocks.

    Parameters
    ----------
    inplanes : int
    planes : int
    reduction : int, default=1
    stride : int, default=1
    downsample : nn.Module or None, default=None

    Example
    -------
    >>> blk = SEBasicBlock(128, 256, reduction=4, stride=2,
    ...                    downsample=nn.Sequential(conv1x1(128, 256, 2), nn.BatchNorm2d(256)))
    >>> x = torch.randn(2, 128, 200, 81)
    >>> blk(x).shape
    torch.Size([2, 256, 100, 41])
    """

    def __init__(self, inplanes, planes, reduction=1, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()

        self.activation = nn.ReLU()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(planes, reduction)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class BasicBlock(nn.Module):
    """
    Pre-activation residual block (no SE).

    Parameters
    ----------
    inplanes : int
    planes : int
    stride : int, default=1
    downsample : nn.Module or None, default=None

    Example
    -------
    >>> blk = BasicBlock(256, 256)
    >>> x = torch.randn(2, 256, 80, 41)
    >>> blk(x).shape
    torch.Size([2, 256, 80, 41])
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.activation = nn.ReLU()

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
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class SEKiwanoBasicBlock(nn.Module):
    """
    Pre-activation residual block with SE attention.

    Identical to `SEBasicBlock` in computation; named variant for
    stacks that interleave SE and non-SE blocks.

    Parameters
    ----------
    inplanes : int
    planes : int
    reduction : int, default=1
    stride : int, default=1
    downsample : nn.Module or None, default=None

    Example
    -------
    >>> blk = SEKiwanoBasicBlock(128, 256, reduction=4, stride=2,
    ...                    downsample=nn.Sequential(conv1x1(128, 256, 2), nn.BatchNorm2d(256)))
    >>> x = torch.randn(2, 128, 200, 81)
    >>> blk(x).shape
    torch.Size([2, 256, 100, 41])
    """

    def __init__(self, inplanes, planes, reduction=1, stride=1, downsample=None):
        super(SEKiwanoBasicBlock, self).__init__()

        self.activation = nn.SiLU()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(planes, reduction)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class KiwanoBasicBlock(nn.Module):
    """
    Pre-activation residual block (no SE).

    Same ordering as `SEKiwanoBasicBlock` (BN → SiLU → Conv) but without
    the SE attention module.

    Parameters
    ----------
    inplanes : int
        Input channels.
    planes : int
        Output channels.
    stride : int, default=1
        Stride for the first 3×3 conv.
    downsample : nn.Module or None, default=None
        Optional residual projection.

    Example
    -------
    >>> blk = KiwanoBasicBlock(256, 256)
    >>> x = torch.randn(4, 256, 120, 81)
    >>> blk(x).shape
    torch.Size([4, 256, 120, 81])
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(KiwanoBasicBlock, self).__init__()

        self.activation = nn.SiLU()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class SubCenterAMSMLoss(nn.Module):
    """
    Additive Margin Softmax with sub-centers (SubCenter-AM-Softmax).

    Uses K sub-centers per class to model intra-class variability. During the
    forward pass, similarities to sub-centers are computed and the maximum
    over sub-centers is taken per class before applying margin m and scale s.

    Parameters
    ----------
    num_features : int
        Embedding dimensionality.
    num_classes : int
        Number of training speaker identities.
    s : float, default=30.0
        Scale applied to logits.
    m : float, default=0.4
        Additive angular margin subtracted from target-class logit.
    k : int, default=3
        Number of sub-centers per class.

    Example
    -------
    >>> crit = SubCenterAMSMLoss(256, 6000, s=30.0, m=0.3, k=3)
    >>> x = torch.randn(32, 256)
    >>> y = torch.randint(0, 6000, (32,))
    >>> logits = crit(x, y)  # pass to CrossEntropyLoss if needed
    >>> logits.shape
    torch.Size([32, 6000])
    """

    def __init__(self, num_features, num_classes, s=30.0, m=0.4, k=3):
        super(SubCenterAMSMLoss, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes

        self.s = s
        self.m = m
        self.k = k

        self.W = nn.Parameter(torch.FloatTensor(self.k, num_features, num_classes))

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

    def forward(self, input, label):
        x = F.normalize(input).unsqueeze(0).expand(self.k, *input.shape)
        W = F.normalize(self.W, dim=1)
        logits = torch.bmm(x, W)
        logits = torch.max(logits, dim=0)[0]

        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot

        # feature re-scale
        output *= self.s
        return output


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


class PreResNet(nn.Module):
    """
    PreResNet: Residual convolutional feature extractor for speaker embedding backbones.

    This module implements a ResNet-style convolutional front-end with Squeeze-and-Excitation (SE)
    blocks in the early stages. It operates directly on time-frequency representations such as
    log-mel spectrograms and produces a high-level feature map suitable for subsequent temporal
    pooling (e.g., attentive statistics pooling).

    The architecture uses a shallow stem convolution followed by four residual stages, where
    the first two stages use SE-enhanced residual blocks (`SEBasicBlock`) and the latter two
    use standard residual blocks (`BasicBlock`). Each stage can perform optional downsampling
    via strided convolution to reduce temporal and frequency resolution progressively.

    Parameters
    ----------
    channels : list[int], default=[128, 128, 256, 256]
        Number of convolutional channels in each of the four residual stages.
    num_blocks : list[int], default=[3, 8, 18, 3]
        Number of residual blocks in each stage.

    Example
    -------
    >>> import torch
    >>> from model import PreResNet
    >>>
    >>> # Create the model
    >>> model = PreResNet(channels=[128, 128, 256, 256], num_blocks=[3, 8, 18, 3])
    >>>
    >>> # Input: batch of log-mel spectrograms
    >>> x = torch.randn(16, 1, 300, 81)  # (batch, channel, time, freq)
    >>>
    >>> # Forward pass
    >>> features = model(x)
    >>> features.shape
    torch.Size([16, 256 * 11, time'])  # Example: depends on downsampling
    """

    def __init__(self, channels=[128, 128, 256, 256], num_blocks=[3, 8, 18, 3]):
        super(PreResNet, self).__init__()

        self.channels = channels
        self.num_blocks = num_blocks

        self.pre_conv1 = nn.Conv2d(1, channels[0], 3, 1, 1, bias=False)
        self.pre_bn1 = nn.BatchNorm2d(channels[0])
        self.pre_activation1 = nn.SiLU()

        self.layer1 = self._make_layer_se(
            channels[0], channels[0], num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer_se(
            channels[0], channels[1], num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            channels[1], channels[2], num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            channels[2], channels[3], num_blocks[3], stride=2
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_se(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )

        layers = []
        layers.append(SEBasicBlock(inchannel, outchannel, 1, stride, downsample))

        for i in range(1, block_num):
            layers.append(SEBasicBlock(outchannel, outchannel, 1))
        return nn.Sequential(*layers)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )

        layers = []
        layers.append(BasicBlock(inchannel, outchannel, stride, downsample))

        for i in range(1, block_num):
            layers.append(BasicBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x, iden=None):
        x = self.pre_conv1(x)
        x = self.pre_bn1(x)
        x = self.pre_activation1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(2, 3)
        x = x.flatten(1, 2)

        return x


class KiwanoPreResNet(nn.Module):
    """
    KiwanoPreResNet: Residual convolutional feature extractor for speaker embedding backbones.

    This module implements a ResNet-style convolutional front-end with Squeeze-and-Excitation (SE)
    blocks in the early stages. It operates directly on time-frequency representations such as
    log-mel spectrograms and produces a high-level feature map suitable for subsequent temporal
    pooling (e.g., attentive statistics pooling).

    The architecture uses a shallow stem convolution followed by four residual stages, where
    the first two stages use SE-enhanced residual blocks (`SEKiwanoBasicBlock`) and the latter two
    use standard residual blocks (`KiwanoBasicBlock`). Each stage can perform optional downsampling
    via strided convolution to reduce temporal and frequency resolution progressively.

    Parameters
    ----------
    channels : list[int], default=[128, 128, 256, 256]
        Number of convolutional channels in each of the four residual stages.
    num_blocks : list[int], default=[3, 8, 18, 3]
        Number of residual blocks in each stage.

    Example
    -------
    >>> import torch
    >>> from model import KiwanoPreResNet
    >>>
    >>> # Create the model
    >>> model = KiwanoPreResNet(channels=[128, 128, 256, 256], num_blocks=[3, 8, 18, 3])
    >>>
    >>> # Input: batch of log-mel spectrograms
    >>> x = torch.randn(16, 1, 300, 81)  # (batch, channel, time, freq)
    >>>
    >>> # Forward pass
    >>> features = model(x)
    >>> features.shape
    torch.Size([16, 256 * 11, time'])  # Example: depends on downsampling
    """

    def __init__(self, channels=[128, 128, 256, 256], num_blocks=[3, 8, 18, 3]):
        super(KiwanoPreResNet, self).__init__()

        self.channels = channels
        self.num_blocks = num_blocks

        self.pre_conv1 = nn.Conv2d(1, channels[0], 3, 1, 1, bias=False)
        self.pre_bn1 = nn.BatchNorm2d(channels[0])
        self.pre_activation1 = nn.SiLU()

        self.layer1 = self._make_layer_se(
            channels[0], channels[0], num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer_se(
            channels[0], channels[1], num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            channels[1], channels[2], num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            channels[2], channels[3], num_blocks[3], stride=2
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_se(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )

        layers = []
        layers.append(SEKiwanoBasicBlock(inchannel, outchannel, 1, stride, downsample))

        for i in range(1, block_num):
            layers.append(SEKiwanoBasicBlock(outchannel, outchannel, 1))
        return nn.Sequential(*layers)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )

        layers = []
        layers.append(KiwanoBasicBlock(inchannel, outchannel, stride, downsample))

        for i in range(1, block_num):
            layers.append(KiwanoBasicBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x, iden=None):
        x = self.pre_conv1(x)
        x = self.pre_bn1(x)
        x = self.pre_activation1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(2, 3)
        x = x.flatten(1, 2)

        return x


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


class ResNet(nn.Module):
    """
    ResNet: A residual convolutional backbone with attentive pooling and
    margin-based classification head for speaker verification.

    This architecture follows a typical modern speaker embedding pipeline:
    a convolutional front-end (PreResNet) extracts high-level time-frequency
    representations, an attentive statistics pooling (ASTP) aggregates temporal
    information, and a projection head (SpeakerEmbedding) maps to a fixed-dimensional
    speaker embedding space. During training, an additive margin softmax (AMSMLoss)
    is used for discriminative supervision over speaker identities.

    Parameters
    ----------
    input_features : int, default=81
        Number of input features per frame (e.g., filterbank or MFCC dimension).
    embed_features : int, default=256
        Dimensionality of the output speaker embedding vector.
    num_classes : int, default=6000
        Number of speaker identities in the training set (for classification loss).
    channels : list[int], default=[128, 128, 256, 256]
        Number of output channels for each residual stage in the backbone.
    num_blocks : list[int], default=[3, 8, 18, 3]
        Number of residual blocks in each stage.

    Example
    -------
    >>> import torch
    >>> from model import ResNet
    >>>
    >>> # Create model
    >>> model = ResNet(input_features=81, embed_features=256, num_classes=6000)
    >>>
    >>> # Batch of log-mel spectrograms (batch=8, channels=1, time=300, freq=81)
    >>> x = torch.randn(8, 1, 300, 81)
    >>> speaker_ids = torch.randint(0, 6000, (8,))
    >>>
    >>> # ----- Inference / Enrollment -----
    >>> model.eval()
    >>> with torch.no_grad():
    ...     emb = model(x)             # Returns embeddings (8, 256)
    ...     emb = torch.nn.functional.normalize(emb)
    >>>
    >>> # Cosine similarity for verification
    >>> score = torch.matmul(emb[0], emb[1].T)
    """

    def __init__(
        self,
        input_features=81,
        embed_features=256,
        num_classes=6000,
        channels=[128, 128, 256, 256],
        num_blocks=[3, 8, 18, 3],
    ):
        super(ResNet, self).__init__()

        self.embed_features = embed_features
        self.num_classes = num_classes
        self.channels = channels
        self.num_blocks = num_blocks

        self.preresnet = PreResNet(self.channels, self.num_blocks)

        self.temporal_pooling = ASTP(channels[3] * 11, channels[3] // 2)

        self.embedding = SpeakerEmbedding(2 * 11 * channels[3], self.embed_features)

        self.output = AMSMLoss(self.embed_features, self.num_classes)

    def extra_repr(self):
        return "embed_features={}, num_classes={}".format(
            self.embed_features, self.num_classes
        )

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

    def get_top_training_speaker(self, x):
        x = self.preresnet(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        x = F.normalize(x)

        W = F.normalize(self.output.get_w())

        logits = F.linear(x, W)

        softmax_tensor = torch.nn.functional.softmax(logits, dim=1)

        sorted_tensor, _ = torch.sort(softmax_tensor, descending=True)

        cumulative_sum = torch.cumsum(sorted_tensor, dim=1)

        counts = x.shape[1] - (cumulative_sum > 0.8).float().argmax(dim=1) + 1

        return counts

    def forward(self, x, iden=None):
        x = self.preresnet(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x


class KiwanoResNet(nn.Module):
    """
    KiwanoResNet: A residual convolutional backbone with attentive pooling and
    margin-based classification head for speaker verification.

    This architecture follows a typical modern speaker embedding pipeline:
    a convolutional front-end (PreResNet) extracts high-level time-frequency
    representations, an attentive statistics pooling (ASTP) aggregates temporal
    information, and a projection head (SpeakerEmbedding) maps to a fixed-dimensional
    speaker embedding space. During training, an additive margin softmax (AMSMLoss)
    is used for discriminative supervision over speaker identities.

    Parameters
    ----------
    input_features : int, default=81
        Number of input features per frame (e.g., filterbank or MFCC dimension).
    embed_features : int, default=256
        Dimensionality of the output speaker embedding vector.
    num_classes : int, default=6000
        Number of speaker identities in the training set (for classification loss).
    channels : list[int], default=[128, 128, 256, 256]
        Number of output channels for each residual stage in the backbone.
    num_blocks : list[int], default=[3, 8, 18, 3]
        Number of residual blocks in each stage.

    Example
    -------
    >>> import torch
    >>> from model import KiwanoResNet
    >>>
    >>> # Create model
    >>> model = KiwanoResNet(input_features=81, embed_features=256, num_classes=6000)
    >>>
    >>> # Batch of log-mel spectrograms (batch=8, channels=1, time=300, freq=81)
    >>> x = torch.randn(8, 1, 300, 81)
    >>> speaker_ids = torch.randint(0, 6000, (8,))
    >>>
    >>> # ----- Inference / Enrollment -----
    >>> model.eval()
    >>> with torch.no_grad():
    ...     emb = model(x)             # Returns embeddings (8, 256)
    ...     emb = torch.nn.functional.normalize(emb)
    >>>
    >>> # Cosine similarity for verification
    >>> score = torch.matmul(emb[0], emb[1].T)
    """

    def __init__(
        self,
        input_features=81,
        embed_features=256,
        num_classes=6000,
        channels=[128, 128, 256, 256],
        num_blocks=[3, 8, 18, 3],
    ):
        super(KiwanoResNet, self).__init__()

        self.embed_features = embed_features
        self.num_classes = num_classes
        self.channels = channels
        self.num_blocks = num_blocks

        self.preresnet = KiwanoPreResNet(self.channels, self.num_blocks)

        self.temporal_pooling = ASTP(channels[3] * 11, channels[3] // 2)

        self.embedding = SpeakerEmbedding(2 * 11 * channels[3], self.embed_features)

        self.output = AMSMLoss(self.embed_features, self.num_classes)

    def extra_repr(self):
        return "embed_features={}, num_classes={}".format(
            self.embed_features, self.num_classes
        )

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

    def forward(self, x, iden=None):
        x = self.preresnet(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x
