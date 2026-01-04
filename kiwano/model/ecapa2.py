import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SubCenterAAMSoftmaxLoss_intertopk(nn.Module):
    def __init__(self, in_features, num_classes, margin=0.2, scale=30.0, k_subcenters=3, mp=0.06, k_top=5):
        super(SubCenterAAMSoftmaxLoss_intertopk, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.k = k_subcenters

        self.weight = nn.Parameter(torch.randn(num_classes * k_subcenters, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)

    def get_m(self):
        return self.margin

    def get_s(self):
        return self.scale

    def set_m(self, m):
        self.margin = m

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.mmm = 1.0 + math.cos(math.pi - self.margin)

        if self.margin > 0.001:
            mp = self.mp * (self.margin / 0.2)
        else:
            mp = 0.0

        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)


    def set_s(self, s):
        self.scale = s


    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        weights = F.normalize(self.weight)

        cosine_sim = F.linear(embeddings, weights)

        if labels == None:
            return cosine_sim

        cosine_sim = cosine_sim.view(-1, self.num_classes, self.k)

        cosine_sim, _ = torch.max(cosine_sim, dim=2)

        sine_sim = torch.sqrt(1.0 - torch.clamp(cosine_sim ** 2, 0, 1))

        phi = cosine_sim * self.cos_m - sine_sim * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        phi = torch.where(cosine_sim > self.th, phi, cosine_sim - self.mmm)

        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        _, top_k_index = torch.topk(cosine - 2 * one_hot, self.k_top)
        top_k_one_hot = input.new_zeros(cosine_sim.size()).scatter_(1, top_k_index, 1)


        output = (one_hot * phi) + (top_k_one_hot * phi_mp) + ((1.0 - one_hot - top_k_one_hot) * cosine)
        #output = one_hot * phi + (1.0 - one_hot) * cosine_sim

        output *= self.scale

        return output







class SubCenterAAMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, num_classes, margin=0.2, scale=30.0, k_subcenters=3):
        super(SubCenterAAMSoftmaxLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.k = k_subcenters

        self.weight = nn.Parameter(torch.randn(num_classes * k_subcenters, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def get_m(self):
        return self.margin

    def get_s(self):
        return self.scale

    def set_m(self, m):
        self.margin = m

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin


    def set_s(self, s):
        self.scale = s


    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        weights = F.normalize(self.weight)

        cosine_sim = F.linear(embeddings, weights)

        if labels == None:
            return cosine_sim

        cosine_sim = cosine_sim.view(-1, self.num_classes, self.k)

        cosine_sim, _ = torch.max(cosine_sim, dim=2)

        sine_sim = torch.sqrt(1.0 - torch.clamp(cosine_sim ** 2, 0, 1))

        phi = cosine_sim * self.cos_m - sine_sim * self.sin_m

        phi = torch.where(cosine_sim > self.th, phi, cosine_sim - self.mm)

        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = one_hot * phi + (1.0 - one_hot) * cosine_sim


        output *= self.scale

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

    def get_w(self):
        return self.W

    def forward(self, input, label = None):
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





class ECAPA2EmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ECAPA2EmbeddingBlock, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)





class ECAPA2AttentiveStatPoolingBlock(nn.Module):
    """Attentive statistic pooling module."""
    def __init__(self, in_channels, attention_channels=128, out_channels=3072):
        super(ECAPA2AttentiveStatPoolingBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels*3, attention_channels, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, in_channels, kernel_size=1, bias=True),
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        mean_x = torch.mean(x, dim=2, keepdim=True)
        mean_x = mean_x.expand_as(x)

        var_x = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=True) + 1e-05)
        std_x = var_x.expand_as(x)

        concatenated = torch.cat([x, mean_x, std_x], dim=1)
        attention_weights_logits = self.attention(concatenated)
        attention_weights = torch.nn.functional.softmax(attention_weights_logits, dim=2)
        x_scaled = x*attention_weights
        mu = torch.sum(x_scaled, dim=2)

        sum_squared = torch.sum(torch.mul(x_scaled, x), dim=2)
        variance_estimate = torch.abs(sum_squared - torch.pow(mu, 2))
        sigma = torch.sqrt(torch.add(variance_estimate, 1e-05))
        pool = torch.cat([mu, sigma], dim=1)

        return self.bn(pool)



class ECAPA2SEBlock(nn.Module):
    """Squeeze–excitation block for 2D feature maps."""
    def __init__(self, in_channels, out_channels):
        super(ECAPA2SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, in_channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        y = torch.mean(x, dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.bn1(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y_unsq = y.unsqueeze(2)
        y_expanded = y_unsq.expand_as(x)
        return x * y_expanded




class ECAPA2Res2NetConv1d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=2, dilation=2):
        super(ECAPA2Res2NetConv1d, self).__init__()
        self.n_groups = 8

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv_filters = nn.ModuleList([
            nn.Conv1d(channels // self.n_groups, channels // self.n_groups, kernel_size, padding=padding, dilation=dilation, padding_mode='reflect', bias=True)
            for _ in range(self.n_groups - 1)
            ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(channels // self.n_groups)
            for _ in range(self.n_groups - 1)
            ])

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(channels)

        self.se = ECAPA2SEBlock(channels, 128)


    def forward(self, x):
        x = self.bn1(self.relu1(self.conv1(x)))

        chunks = torch.chunk(x, self.n_groups, dim=1)
        outputs = []
        outputs.append(chunks[0])
        for i in range(1, self.n_groups):
            if i == 1:
                sp = self.conv_filters[i - 1](chunks[i])
            else:
                sp = self.conv_filters[i - 1](chunks[i] + outputs[i - 1])
            sp = F.relu(sp)
            sp = self.batch_norms[i - 1](sp)
            outputs.append(sp)

        x = torch.cat(outputs, dim=1)

        x = self.bn2(self.relu2(self.conv2(x)))

        x = self.se(x)

        return x





class ECAPA2DenseBlock(nn.Module):
    """A basic 1D TDNN block."""
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(ECAPA2DenseBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))




class ECAPA2TDNNBlock(nn.Module):
    """A basic 1D TDNN block."""
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1):
        super(ECAPA2TDNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))


class ECAPA2DownsampleBlock(nn.Module):
    """Downsampling block: a 1x1 convolution followed by batch normalization."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(ECAPA2DownsampleBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class ECAPA2SEBlock2d(nn.Module):
    """Squeeze–excitation block for 2D feature maps."""
    def __init__(self, in_channels, out_channels):
        super(ECAPA2SEBlock2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, in_channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view([b, c])
        w = self.fc1(w)
        w = self.relu(w)
        w = self.bn1(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        w = w.view([b, c, 1, 1])
        return x * w



class ECAPA2ConvBlock(nn.Module):
    """A 2D convolutional block with three 3x3 conv layers, ReLU, BatchNorm and SE attention."""
    def __init__(self, in_channels, out_channels, stride=(1,1)):
        super(ECAPA2ConvBlock, self).__init__()
        self.stride=stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=self.stride, bias=True)
        self.relu1 = nn.ReLU()
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU()
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.se = ECAPA2SEBlock2d(out_channels, 128)


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.bn3(out)

        out = self.se(out)

        return out


class PreECAPA2V2(nn.Module):
    def __init__(self):
        super(PreECAPA2V2, self).__init__()

        self.downsample_1 = ECAPA2DownsampleBlock(164, 164, stride=(2,1))
        self.downsample_2 = ECAPA2DownsampleBlock(164, 164, stride=(2,1))
        self.downsample_3 = ECAPA2DownsampleBlock(192, 192, stride=(2,1))
        self.downsample_4 = ECAPA2DownsampleBlock(192, 192, stride=(2,1))

        self.conv_1 = ECAPA2ConvBlock(1, 164)
        self.conv_2 = ECAPA2ConvBlock(164, 164)
        self.conv_3 = ECAPA2ConvBlock(164, 164)

        self.conv_4 = ECAPA2ConvBlock(164, 164, stride=(2,1))
        self.conv_5 = ECAPA2ConvBlock(164, 164)
        self.conv_6 = ECAPA2ConvBlock(164, 164)
        self.conv_7 = ECAPA2ConvBlock(164, 164)

        self.conv_8 = ECAPA2ConvBlock(164, 164, stride=(2,1))
        self.conv_9 = ECAPA2ConvBlock(164, 164)
        self.conv_10 = ECAPA2ConvBlock(164, 192)
        self.conv_11 = ECAPA2ConvBlock(192, 192)

        self.conv_12 = ECAPA2ConvBlock(192, 192, stride=(2,1))
        self.conv_13 = ECAPA2ConvBlock(192, 192)
        self.conv_14 = ECAPA2ConvBlock(192, 192)
        self.conv_15 = ECAPA2ConvBlock(192, 192)

        self.conv_16 = ECAPA2ConvBlock(192, 192, stride=(2,1))
        self.conv_17 = ECAPA2ConvBlock(192, 192)
        self.conv_18 = ECAPA2ConvBlock(192, 192)
        self.conv_19 = ECAPA2ConvBlock(192, 192)
        self.conv_20 = ECAPA2ConvBlock(192, 192)

        self.tdnn_1 = ECAPA2TDNNBlock(3072, 1024, kernel_size=1)
        self.tdnn_2 = ECAPA2Res2NetConv1d(1024)

        self.dense_1 = ECAPA2DenseBlock(1024, 1536)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x) + x
        x = self.conv_3(x) + x

        x = self.conv_4(x) + self.downsample_1(x)
        x = self.conv_5(x) + x
        x = self.conv_6(x) + x
        x = self.conv_7(x) + x

        x = self.conv_8(x) + self.downsample_2(x)
        x = self.conv_9(x) + x
        x = self.conv_10(x)
        x = self.conv_11(x) + x

        x = self.conv_12(x) + self.downsample_3(x)
        x = self.conv_13(x) + x
        x = self.conv_14(x) + x
        x = self.conv_15(x) + x

        x = self.conv_16(x) + self.downsample_4(x)
        x = self.conv_17(x) + x
        x = self.conv_18(x) + x
        x = self.conv_19(x) + x
        x = self.conv_20(x) + x

        x = x.flatten(1, 2)

        x = self.tdnn_1(x)
        x = self.tdnn_2(x) + x

        x = self.dense_1(x)

        return x


class ECAPA2V3SubCenterV3(nn.Module):
    def __init__(self, num_classes=6000, emb=256):
        super(ECAPA2V3SubCenterV3, self).__init__()

        self.preecapa2 = PreECAPA2V2()

        self.temporal_pooling = ECAPA2AttentiveStatPoolingBlock(1536)

        self.embedding = ECAPA2EmbeddingBlock(1536*2, emb)

        self.norm_embed = torch.nn.BatchNorm1d(emb)

        self.output = SubCenterAAMSoftmaxLoss(emb, num_classes, margin=0.2, scale=32.0, k_subcenters=2)


    def forward(self, x, iden=None):

        x = self.preecapa2(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.norm_embed(x)

        x = self.output(x, iden)

        return x

    def extra_repr(self):
        return ""

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)






class ECAPA2V3SubCenterV2(nn.Module):
    def __init__(self, num_classes=6000, emb=256):
        super(ECAPA2V3SubCenterV2, self).__init__()

        self.preecapa2 = PreECAPA2V2()

        self.temporal_pooling = ECAPA2AttentiveStatPoolingBlock(1536)

        self.embedding = ECAPA2EmbeddingBlock(1536*2, emb)

        self.output = SubCenterAAMSoftmaxLoss(emb, num_classes, margin=0.2, scale=32.0, k_subcenters=2)


    def forward(self, x, iden=None):

        x = self.preecapa2(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x

    def extra_repr(self):
        return ""

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)





class ECAPA2V3SubCenter(nn.Module):
    def __init__(self, num_classes=6000, emb=256):
        super(ECAPA2V3SubCenter, self).__init__()

        self.preecapa2 = PreECAPA2V2()

        self.temporal_pooling = ECAPA2AttentiveStatPoolingBlock(1536)

        self.embedding = SpeakerEmbedding(1536*2, emb)

        self.output = SubCenterAAMSoftmaxLoss(emb, num_classes, margin=0.2, scale=32.0, k_subcenters=2)


    def forward(self, x, iden=None):

        x = self.preecapa2(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x

    def extra_repr(self):
        return ""

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)



class ECAPA2V3(nn.Module):
    def __init__(self, num_classes=6000):
        super(ECAPA2V3, self).__init__()

        self.preecapa2 = PreECAPA2V2()

        self.temporal_pooling = ECAPA2AttentiveStatPoolingBlock(1536)

        self.embedding = SpeakerEmbedding(1536*2, 192)

        self.output = AMSMLoss(192, num_classes, s=30, m=0.2)


    def forward(self, x, iden=None):

        x = self.preecapa2(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x

    def extra_repr(self):
        return ""

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)







class ECAPA2V2(nn.Module):
    def __init__(self):
        super(ECAPA2V2, self).__init__()

        self.preecapa2 = PreECAPA2V2()

        self.temporal_pooling = ECAPA2AttentiveStatPoolingBlock(1536)

        self.embedding = ECAPA2EmbeddingBlock(1536*2, 192)

        self.output = AMSMLoss(192, 6000, s=30, m=0.2)


    def forward(self, x, iden=None):

        x = self.preecapa2(x)

        x = self.temporal_pooling(x)

        x = self.embedding(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x

    def extra_repr(self):
        return ""

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)









class ECAPA2(nn.Module):
    def __init__(self):
        super(ECAPA2, self).__init__()

        self.downsample_1 = ECAPA2DownsampleBlock(164, 164, stride=(2,1))
        self.downsample_2 = ECAPA2DownsampleBlock(164, 164, stride=(2,1))
        self.downsample_3 = ECAPA2DownsampleBlock(192, 192, stride=(2,1))
        self.downsample_4 = ECAPA2DownsampleBlock(192, 192, stride=(2,1))

        self.conv_1 = ECAPA2ConvBlock(1, 164)
        self.conv_2 = ECAPA2ConvBlock(164, 164)
        self.conv_3 = ECAPA2ConvBlock(164, 164)

        self.conv_4 = ECAPA2ConvBlock(164, 164, stride=(2,1))
        self.conv_5 = ECAPA2ConvBlock(164, 164)
        self.conv_6 = ECAPA2ConvBlock(164, 164)
        self.conv_7 = ECAPA2ConvBlock(164, 164)

        self.conv_8 = ECAPA2ConvBlock(164, 164, stride=(2,1))
        self.conv_9 = ECAPA2ConvBlock(164, 164)
        self.conv_10 = ECAPA2ConvBlock(164, 192)
        self.conv_11 = ECAPA2ConvBlock(192, 192)

        self.conv_12 = ECAPA2ConvBlock(192, 192, stride=(2,1))
        self.conv_13 = ECAPA2ConvBlock(192, 192)
        self.conv_14 = ECAPA2ConvBlock(192, 192)
        self.conv_15 = ECAPA2ConvBlock(192, 192)

        self.conv_16 = ECAPA2ConvBlock(192, 192, stride=(2,1))
        self.conv_17 = ECAPA2ConvBlock(192, 192)
        self.conv_18 = ECAPA2ConvBlock(192, 192)
        self.conv_19 = ECAPA2ConvBlock(192, 192)
        self.conv_20 = ECAPA2ConvBlock(192, 192)

        self.tdnn_1 = ECAPA2TDNNBlock(3072, 1024, kernel_size=1)
        self.tdnn_2 = ECAPA2Res2NetConv1d(1024)

        self.dense_1 = ECAPA2DenseBlock(1024, 1536)

        self.pooling_1 = ECAPA2AttentiveStatPoolingBlock(1536)

        self.dense_2 = ECAPA2EmbeddingBlock(1536*2, 192)

        self.output = AMSMLoss(192, 6000, s=30, m=0.2)



    def forward(self, x, iden=None):

        x = self.conv_1(x)
        x = self.conv_2(x) + x
        x = self.conv_3(x) + x

        x = self.conv_4(x) + self.downsample_1(x)
        x = self.conv_5(x) + x
        x = self.conv_6(x) + x
        x = self.conv_7(x) + x

        x = self.conv_8(x) + self.downsample_2(x)
        x = self.conv_9(x) + x
        x = self.conv_10(x)
        x = self.conv_11(x) + x

        x = self.conv_12(x) + self.downsample_3(x)
        x = self.conv_13(x) + x
        x = self.conv_14(x) + x
        x = self.conv_15(x) + x

        x = self.conv_16(x) + self.downsample_4(x)
        x = self.conv_17(x) + x
        x = self.conv_18(x) + x
        x = self.conv_19(x) + x
        x = self.conv_20(x) + x

        x = x.flatten(1, 2)

        x = self.tdnn_1(x)
        x = self.tdnn_2(x) + x

        x = self.dense_1(x)

        x = self.pooling_1(x)

        x = self.dense_2(x)

        if iden == None:
            return x

        x = self.output(x, iden)

        return x

    def extra_repr(self):
        return ""

    def get_m(self):
        return self.output.get_m()

    def get_s(self):
        return self.output.get_s()

    def set_m(self, m):
        self.output.set_m(m)

    def set_s(self, s):
        self.output.set_s(s)

