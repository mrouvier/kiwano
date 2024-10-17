import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from .WavLM_v2 import *

from conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones

import torch
import torch.nn as nn




class MyConformer(nn.Module):
    def __init__(self, emb_size=192, heads=4, ffmult=4, exp_fac=2, kernel_size=32, n_encoders=4):
        super(MyConformer, self).__init__()
        self.dim_head=int(emb_size/heads)
        self.dim=emb_size
        self.heads=heads
        self.kernel_size=kernel_size
        self.n_encoders=n_encoders
        self.encoder_blocks=_get_clones( ConformerBlock( dim = emb_size, dim_head=self.dim_head, heads=heads, ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size), n_encoders)
        self.class_token = nn.Parameter(torch.rand(1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)

    def forward(self, x):
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])
        for layer in self.encoder_blocks:
            x = layer(x)
        embedding=x[:,0,:]

        out = self.fc5(embedding)
        return out


class Conformer(nn.Module):
    def __init__(self, name):
        super(Conformer, self).__init__()
        checkpoint = torch.load(name)
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint['model'])

        self.LL = nn.Linear(1024, 192)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer = MyConformer()


    def get_w(self):
        return self.output.get_w()

    def forward(self, wav_and_flag):
        x = wav_and_flag
        layer, res =  self.model.extract_features(x)

        x = self.LL(layer)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out = self.conformer(x)

        return out


    def get_m(self):
        return 0


    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            

            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);



