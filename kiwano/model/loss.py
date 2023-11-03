import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

from kiwano.model.tools import accuracy


'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, loss_margin, loss_scale):
        super(AAMsoftmax, self).__init__()
        self.loss_margin = loss_margin
        self.loss_scale = loss_scale
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.loss_margin)
        self.sin_m = math.sin(self.loss_margin)
        self.th = math.cos(math.pi - self.loss_margin)
        self.mm = math.sin(math.pi - self.loss_margin) * self.loss_margin

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.loss_scale

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1


class JeffreysLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', coeff1=0.0, coeff2=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, coeff1=0.0, coeff2=0.0):
        assert 0 <= coeff1 < 1
        assert 0 <= coeff2 < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device).fill_(coeff1 / (n_classes - 1)).scatter_(1, targets.data.unsqueeze(1), 1. - coeff1-coeff2)
        return targets

    @staticmethod
    def _jeffreys_one_cold(targets: torch.Tensor, n_classes: int):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device).fill_(1).scatter_(1, targets.data.unsqueeze(1),0.0)
        return targets

    @staticmethod
    def _jeffreys_one_hot(targets: torch.Tensor, n_classes: int):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device).fill_(0).scatter_(1, targets.data.unsqueeze(1),1.0)
        return targets

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        targets1 = JeffreysLoss._smooth_one_hot(targets, inputs.size(-1), self.coeff1,self.coeff2)
        sm = F.softmax(inputs, -1)
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        # (cross-entropy)  & (Jeffreys part 1=label-smoothing)
        loss = -(targets1 * lsm).sum(-1)

        # Jeffreys part 2
        lsmsm = lsm * sm
        targets21 = JeffreysLoss._jeffreys_one_cold(targets, inputs.size(-1),)
        loss1 = (targets21 * lsmsm).sum(-1)

        targets22 = JeffreysLoss._jeffreys_one_hot(targets, inputs.size(-1),)
        loss2 = (targets22 * sm).sum(-1)

        loss3 = loss1/(torch.ones_like(loss2)-loss2)

        loss3 *= self.coeff2
        loss = loss + loss3
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss




