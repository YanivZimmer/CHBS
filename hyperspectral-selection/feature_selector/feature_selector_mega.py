import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from feature_selector.feature_selector_gumble import FeatureSelectorGumble
from feature_selector.concrete_autoencoder import ConcreteEncoder

class FeatureSelectorMega(nn.Module):
    def __init__(self, input_dim, device,target_number=4, temp=0.01):
        print("here9")
        super(FeatureSelectorMega, self).__init__()
        self.target_number = target_number
        self.device = device
        self.input_dim = input_dim
        #self.sub_gates=[FeatureSelectorGumble(input_dim, device,target_number=1, temp=temp) for _ in range(target_number)]
        self.sub_gates=[ConcreteEncoder(input_dim=input_dim,output_dim=1,device="cuda") for _ in range(target_number)]
        self.mu = None#torch.cat([self.sub_gates[2].mu,self.sub_gates[1].mu],0) #torch.cat([sub.mu.unsqueeze(0) for sub in self.sub_gates], dim=0).sum(dim=0)
        self.mask=None
        
    def get_gates(self, mode):
        return [sub.get_gates(mode) for sub in self.sub_gates]

    def forward(self, x):
        subs_result = [sub.forward(x) for sub in self.sub_gates]
        #assumes bands dim is in 2 dim
        #maybe add sorting?
        subs_cat = torch.cat((self.sub_gates[0](x),self.sub_gates[0](x),self.sub_gates[0](x)), 2)
        #print(subs_result[0].shape)
        #print(subs_cat.shape)
        return subs_cat

    def regularizer(self, x):
        # if self.const_masking is not None:
        #    return torch.Tensor([0])
        """Gaussian CDF."""
        #print(x.shape)
        #print(x[self.last_topk].shape)
        #return 0.5 * (1 + torch.erf(x[self.last_topk] / math.sqrt(2)))
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
