import torch
import torch.nn as nn
import torch.nn.functional as F



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

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
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class AMSMLoss(nn.Module):

    def __init__(self, num_features, num_classes, s=30.0, m=0.4):
        super(AMSMLoss, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)

        # normalize weights
        W = F.normalize(self.W)

        # dot product
        logits = F.linear(x, W)
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot

        # feature re-scale
        output *= self.s
        return output


class ResNet(nn.Module):
    def __init__(self, embed_features=256, num_classes=6000):
        super(ResNet, self).__init__()

        self.embed_features = embed_features
        self.num_classes = num_classes

        self.pre_conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.pre_bn1 = nn.BatchNorm2d(64)
        self.pre_relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 64, 4, stride=2)
        self.layer3 = self._make_layer(64, 128, 6, stride=2)
        self.layer4 = self._make_layer(128, 128, 3, stride=2)

        self.norm_stats = torch.nn.BatchNorm1d(2 * 11 * 128)

        self.fc_embed = nn.Linear(2 * 11 * 128, self.embed_features)
        self.norm_embed = torch.nn.BatchNorm1d(self.embed_features)

        self.attention = nn.Sequential(
            nn.Conv1d(128 * 11, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128 * 11, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.output = AMSMLoss(self.embed_features, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(BasicBlock(inchannel, outchannel, stride, downsample))

        for i in range(1, block_num):
            layers.append(BasicBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x, iden):
        x = self.pre_conv1(x)
        x = self.pre_bn1(x)
        x = self.pre_relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(2, 3)
        x = x.flatten(1, 2)

        w = self.attention(x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
        x = torch.cat([mu, sg], dim=1)
        x = self.norm_stats(x)

        x = self.fc_embed(x)
        x = self.norm_embed(x)

        x = self.output(x, iden)

        return x


