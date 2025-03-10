import math

import numpy as np
import torch
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FeatureSelectorL1Conv(nn.Module):

    def __init__(self,input_channels, lam=1.0, threshold=1e-4):
        in_features, out_features = input_channels, input_channels
        super(FeatureSelectorL1Conv, self).__init__()
        self.threshold = threshold
        self.lam = lam
        self.test = False
        self.mu = torch.nn.Parameter(
            0.01*torch.ones(
                in_features,
            ),
            requires_grad=True,
        )


    def get_gates(self, mode=None):
        return self.mu

    def get_top_gates(self, k=10):
        gates = torch.abs(self.get_gates())
        k_max_val = torch.topk(torch.from_numpy(gates), k).values[k - 1]
        mask = torch.from_numpy(gates) >= k_max_val
        return mask

    def hard_sigmoid(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def regularization(self):
        #TODO make sure you notice p!!
        return 0.005*torch.norm(self.mu, 1)

    def forward(self, x):
        temp = x
        if self.test:
            stochastic_gate = self.get_top_gates()
        else:
            stochastic_gate = self.mu

        if len(x.shape) == 2:
            return x * stochastic_gate
        x = x.squeeze()

        x = torch.transpose(x, 1, 3)
        x = x * stochastic_gate
        x = torch.transpose(x, 1, 3)
        return x.unsqueeze(1)
