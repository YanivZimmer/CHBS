import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .feature_selector_wrapper import FeatureSelectionWrapper


class Baseline(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x


class BaselineFS(Baseline, FeatureSelectionWrapper):
    def __init__(
        self,
        input_channels,
        n_classes,
        dropout=False,
        lam=1,
        sigma=0.5,
        headstart_idx=None,
        device="cuda:0",
    ):
        Baseline.__init__(self, input_channels, n_classes, dropout=dropout)
        FeatureSelectionWrapper.__init__(
            self,
            input_channels,
            sigma=sigma,
            lam=lam,
            device=device,
            headstart_idx=headstart_idx,
        )

    def forward(self, x):
        x = self.feature_selector.forward(x)
        x = Baseline.forward(self=self, x=x)
        return x
