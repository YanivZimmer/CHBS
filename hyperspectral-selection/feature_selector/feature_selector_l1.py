import torch.nn as nn
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DiagonalLinear(nn.Module):

    def __init__(self, in_features, out_features, threshold=1e-4):
        super(DiagonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False,device=device)
        self.mask = torch.eye(in_features, dtype=bool,device=device)
        self.threshold = threshold

    def forward(self, x):
        self.linear.weight.data *= self.mask
        over_threshold = torch.abs(self.linear.weight.data) > self.threshold
        self.linear.weight.data *= over_threshold
        return self.linear(x)
