import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


'''
class SEStoDepthBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, features_per_frames, reduction=1, stride=1, downsample=None, prob=0.5):
        super(SEStoDepthBasicBlock, self).__init__()

        self.activation = nn.SiLU()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(planes, reduction)

        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        if self.training:

            if torch.equal(self.m.sample(), torch.ones(1)):

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
            else:
                residual = x

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual

                return out
        else:
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

'''




class SEBasicBlockX(nn.Module):
    def __init__(self, inplanes, planes, feature_dim, reduction=1, stride=1, downsample=None):
        super(SEBasicBlockX, self).__init__()

        self.pos = nn.Parameter(torch.rand(feature_dim))

        self.activation = nn.SiLU()

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

        x = x + self.pos

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        #out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        
        return out




class BasicBlockX(nn.Module):
    def __init__(self, inplanes, planes, feature_dim, stride=1, downsample=None):
        super(BasicBlockX, self).__init__()

        self.pos = nn.Parameter(torch.rand(feature_dim))

        self.activation = nn.SiLU()

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

        x = x + self.pos

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        #out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        
        return out





class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y





class SEBasicBlockPosition(nn.Module):
    def __init__(self, inplanes, planes, reduction=1, stride=1, downsample=None):
        super(SEBasicBlockPosition, self).__init__()

        self.activation = nn.SiLU()

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
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockPosition, self).__init__()

        self.activation = nn.SiLU()

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
    def __init__(self, inplanes, planes, reduction=1, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()

        self.activation = nn.SiLU()

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
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.activation = nn.SiLU()

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



class SubCenterAMSMLoss(nn.Module):
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
    def __init__(self, input_features=81, embed_features=256, num_classes=6000, channels=[128, 128, 256, 256], num_blocks=[3,8,18,3]):
        super(ResNet, self).__init__()

        self.embed_features = embed_features
        self.num_classes = num_classes
        self.channels = channels
        self.num_blocks = num_blocks

        self.pre_conv1 = nn.Conv2d(1, channels[0], 3, 1, 1, bias=False)
        self.pre_bn1 = nn.BatchNorm2d(channels[0])
        self.pre_activation1 = nn.SiLU()

        self.layer1 = self._make_layer_se(channels[0], channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer_se(channels[0], channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], num_blocks[3], stride=2)

        self.norm_stats = torch.nn.BatchNorm1d(2 * 11 * channels[3])

        self.fc_embed = nn.Linear(2 * 11 * channels[3], self.embed_features)
        self.norm_embed = torch.nn.BatchNorm1d(self.embed_features)

        self.attention = nn.Sequential(
            nn.Conv1d(channels[3] * 11, channels[3]//2, kernel_size=1),
            #nn.SiLU(),
            #nn.BatchNorm1d(channels[3]//2),
            nn.Tanh(),
            nn.Conv1d(channels[3]//2, channels[3] * 11, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.output = AMSMLoss(self.embed_features, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extra_repr(self):
        return 'embed_features={}, num_classes={}'.format(self.embed_features, self.num_classes)

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def _make_layer_se(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
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
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(BasicBlock(inchannel, outchannel, stride, downsample))

        for i in range(1, block_num):
            layers.append(BasicBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x, iden = None):
        x = self.pre_conv1(x)
        x = self.pre_bn1(x)
        x = self.pre_activation1(x)

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

        if iden == None:
            return x

        x = self.output(x, iden)

        return x



class ASTP(nn.Module):
    """
    Attentive statistics pooling: Channel- and context-dependent
    statistics pooling, first used in ECAPA_TDNN.
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
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))

        return torch.cat([mu, sg], dim=1)



class PreResNetV2(nn.Module):
    def __init__(self, input_features=[81, 41, 21, 11], channels=[128, 128, 256, 256], num_blocks=[3,8,18,3]):
        super(PreResNetV2, self).__init__()

        self.channels = channels
        self.num_blocks = num_blocks
        self.input_features = input_features

        self.pre_conv1 = nn.Conv2d(1, channels[0], 3, 1, 1, bias=False)
        self.pre_bn1 = nn.BatchNorm2d(channels[0])
        self.pre_activation1 = nn.SiLU()

        self.layer1 = self._make_layer_se(self.channels[0], self.channels[0], self.input_features[0], self.input_features[0], self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer_se(self.channels[0], self.channels[1], self.input_features[0], self.input_features[1], self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.channels[1], self.channels[2], self.input_features[1], self.input_features[2], self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.channels[2], self.channels[3], self.input_features[2], self.input_features[3], self.num_blocks[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer_se(self, inchannel, outchannel, feature_in, feature_out, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(SEBasicBlockX(inchannel, outchannel, feature_in, 1, stride, downsample))

        for i in range(1, block_num):
            layers.append(SEBasicBlockX(outchannel, outchannel, feature_out, 1))
        return nn.Sequential(*layers)


    def _make_layer(self, inchannel, outchannel, feature_in, feature_out, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(BasicBlockX(inchannel, outchannel, feature_in, stride, downsample))

        for i in range(1, block_num):
            layers.append(BasicBlockX(outchannel, outchannel, feature_out))
        return nn.Sequential(*layers)



    def forward(self, x, iden = None):
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




class PreResNetV3(nn.Module):
    def __init__(self, channels=[128, 128, 256, 256], num_blocks=[3,8,18,3]):
        super(PreResNetV3, self).__init__()

        self.channels = channels
        self.num_blocks = num_blocks

        self.pre_conv1 = nn.Conv2d(1, channels[0], 3, 1, 1, bias=False)
        self.pre_bn1 = nn.BatchNorm2d(channels[0])
        self.pre_activation1 = nn.SiLU()

        self.layer1 = self._make_layer_se(channels[0], channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer_se(channels[0], channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], num_blocks[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_se(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(SEBasicBlockPosition(inchannel, outchannel, 1, stride, downsample))

        for i in range(1, block_num):
            layers.append(SEBasicBlockPosition(outchannel, outchannel, 1))
        return nn.Sequential(*layers)


    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(BasicBlockPosition(inchannel, outchannel, stride, downsample))

        for i in range(1, block_num):
            layers.append(BasicBlockPosition(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x, iden = None):
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





class PreResNet(nn.Module):
    def __init__(self, channels=[128, 128, 256, 256], num_blocks=[3,8,18,3]):
        super(PreResNet, self).__init__()

        self.channels = channels
        self.num_blocks = num_blocks

        self.pre_conv1 = nn.Conv2d(1, channels[0], 3, 1, 1, bias=False)
        self.pre_bn1 = nn.BatchNorm2d(channels[0])
        self.pre_activation1 = nn.SiLU()

        self.layer1 = self._make_layer_se(channels[0], channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer_se(channels[0], channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], num_blocks[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_se(self, inchannel, outchannel, block_num, stride=1):
        downsample = None
        if stride != 1 or inchannel != outchannel:
            downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
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
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(BasicBlock(inchannel, outchannel, stride, downsample))

        for i in range(1, block_num):
            layers.append(BasicBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x, iden = None):
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



class ResNetV3(nn.Module):
    def __init__(self, input_features=81, embed_features=256, num_classes=6000, channels=[128, 128, 256, 256], num_blocks=[3,8,18,3], k=3):
        super(ResNetV3, self).__init__()

        self.embed_features = embed_features
        self.num_classes = num_classes
        self.channels = channels
        self.num_blocks = num_blocks
        self.k = k

        self.preresnet = PreResNet(self.channels, self.num_blocks)

        self.temporal_pooling = ASTP(channels[3] * 11, channels[3]//2)

        self.embedding = SpeakerEmbedding(2*11*channels[3], self.embed_features)

        self.output = SubCenterAMSMLoss(self.embed_features, self.num_classes, k=self.k)

    def extra_repr(self):
        return 'embed_features={}, num_classes={}, k={}'.format(self.embed_features, self.num_classes, self.k)

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

    def forward(self, x, iden = None):
        x = self.preresnet(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x


class ResNetV4b(nn.Module):
    def __init__(self, input_features=81, embed_features=256, num_classes=6000, channels=[128, 128, 256, 256], num_blocks=[3,8,18,3]):
        super(ResNetV4, self).__init__()

        self.input_features = input_features
        self.embed_features = embed_features
        self.num_classes = num_classes
        self.channels = channels
        self.num_blocks = num_blocks

        features = [self.input_features, math.ceil(self.input_features/2), math.ceil(self.input_features/4), math.ceil(self.input_features/8)]

        self.preresnet = PreResNetV3(features, self.channels, self.num_blocks)

        self.temporal_pooling = ASTP(channels[3] * 11, channels[3]//2)

        self.embedding = SpeakerEmbedding(2*11*channels[3], self.embed_features)

        self.output = AMSMLoss(self.embed_features, self.num_classes)

    def extra_repr(self):
        return 'input_features={}, embed_features={}, num_classes={}, channels={}, num_blocks={}'.format(self.input_features, self.embed_features, self.num_classes, " ".join(map(lambda x: str(x), self.channels)), " ".join(map(lambda x: str(x), self.num_blocks))  )

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

    def forward(self, x, iden = None):
        x = self.preresnet(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x





class ResNetV4(nn.Module):
    def __init__(self, input_features=81, embed_features=256, num_classes=6000, channels=[128, 128, 256, 256], num_blocks=[3,8,18,3]):
        super(ResNetV4, self).__init__()

        self.input_features = input_features
        self.embed_features = embed_features
        self.num_classes = num_classes
        self.channels = channels
        self.num_blocks = num_blocks

        features = [self.input_features, math.ceil(self.input_features/2), math.ceil(self.input_features/4), math.ceil(self.input_features/8)]

        self.preresnet = PreResNetV2(features, self.channels, self.num_blocks)

        self.temporal_pooling = ASTP(channels[3] * 11, channels[3]//2)

        self.embedding = SpeakerEmbedding(2*11*channels[3], self.embed_features)

        self.output = AMSMLoss(self.embed_features, self.num_classes)

    def extra_repr(self):
        return 'input_features={}, embed_features={}, num_classes={}, channels={}, num_blocks={}'.format(self.input_features, self.embed_features, self.num_classes, " ".join(map(lambda x: str(x), self.channels)), " ".join(map(lambda x: str(x), self.num_blocks))  )

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

    def forward(self, x, iden = None):
        x = self.preresnet(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x



class ResNetV5(nn.Module):
    def __init__(self, input_features=81, embed_features=256, num_classes=6000, channels=[128, 128, 256, 256], num_blocks=[3,8,18,3]):
        super(ResNetV5, self).__init__()

        self.embed_features = embed_features
        self.num_classes = num_classes
        self.channels = channels
        self.num_blocks = num_blocks

        self.preresnet = PreResNet(self.channels, self.num_blocks)

        self.temporal_pooling = ASTP(channels[3] * 12, channels[3]//2)

        self.embedding = SpeakerEmbedding(2*12*channels[3], self.embed_features)

        self.output = AMSMLoss(self.embed_features, self.num_classes)

    def extra_repr(self):
        return 'embed_features={}, num_classes={}'.format(self.embed_features, self.num_classes)

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

    def forward(self, x, iden = None):
        x = self.preresnet(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x







class ResNetV2(nn.Module):
    def __init__(self, input_features=81, embed_features=256, num_classes=6000, channels=[128, 128, 256, 256], num_blocks=[3,8,18,3]):
        super(ResNetV2, self).__init__()

        self.embed_features = embed_features
        self.num_classes = num_classes
        self.channels = channels
        self.num_blocks = num_blocks

        self.preresnet = PreResNet(self.channels, self.num_blocks)

        self.temporal_pooling = ASTP(channels[3] * 11, channels[3]//2)

        self.embedding = SpeakerEmbedding(2*11*channels[3], self.embed_features)

        self.output = AMSMLoss(self.embed_features, self.num_classes)

    def extra_repr(self):
        return 'embed_features={}, num_classes={}'.format(self.embed_features, self.num_classes)

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

    def forward(self, x, iden = None):
        x = self.preresnet(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x


