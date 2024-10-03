import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers import Wav2Vec2Model, AutoModel, AutoFeatureExtractor
from .WavLM import *

import torch
import torch.nn as nn




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




class MHFATop(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256, number_layers=13):
        super(MHFATop, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(number_layers), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(number_layers), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.number_layers = number_layers

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k)

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2)

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class MHFA(nn.Module):
    def __init__(self, name, number_head=32):
        super(MHFA, self).__init__()
        checkpoint = torch.load(name)
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint['model'])
        self.number_head = number_head

        #self.model = AutoModel.from_pretrained(name)

        self.backend = MHFATop(head_nb=self.number_head)

        self.output = AMSMLoss(256, 2, s=30, m=0.2)


    def forward(self, wav_and_flag, iden = None):
        '''
        out = self.model(wav_and_flag, output_hidden_states=True)

        hidden_states = out.hidden_states

        layer_reps = [x for x in hidden_states]

        x = torch.stack(layer_reps)
        x = x.permute(1, 3, 2, 0)
        '''

        x = wav_and_flag
        cnn_outs, layer_results =  self.model.extract_features(x, output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)

        out = self.backend(x)


        if iden == None:
            return self.output(out)

        out = self.output(out, iden)
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



class MHFA_HF(nn.Module):
    def __init__(self, name):
        super(MHFA_HF, self).__init__()
        self.model = AutoModel.from_pretrained(name)

        self.backend = MHFATop(head_nb=32, number_layers=self.model.config.num_hidden_layers+1, inputs_dim=self.model.config.hidden_size)

        self.output = AMSMLoss(256, 2, s=30, m=0.2)


    def forward(self, wav_and_flag, iden = None):
        out = self.model(wav_and_flag, output_hidden_states=True)

        hidden_states = out.hidden_states

        layer_reps = [x for x in hidden_states]

        x = torch.stack(layer_reps)
        x = x.permute(1, 3, 2, 0)

        out = self.backend(x)

        if iden == None:
            return self.output(out)

        out = self.output(out, iden)
        return out

    def get_m(self):
        return 0



