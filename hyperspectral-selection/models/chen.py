import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from tqdm import tqdm
from feature_selector.feature_selector_general import FeatureSelector
from .feature_selector_wrapper import FeatureSelectionWrapper

class Chen(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """

    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32):
        super(Chen, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (3, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (2, 2, 2))
        #self.pool2 = nn.MaxPool3d((1, 2, 2))
        #self.conv3 = nn.Conv3d(n_planes, n_planes, (2, 2, 2))

        # self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        # self.pool1 = nn.MaxPool3d((1, 2, 2))
        # self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        # self.pool2 = nn.MaxPool3d((1, 2, 2))
        # self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.conv2(x)
            #x = self.pool2(self.conv2(x))
            #x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        #x = self.pool2(x)
        #x = self.dropout(x)
        #x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


class ChenFS(Chen, FeatureSelectionWrapper):
    def __init__(
        self,
        input_channels,
        n_classes,
        patch_size=5,
        lam=1,
        sigma=0.5,
        headstart_idx=None,
        device="cuda:0",
        target_number=10
    ):
        Chen.__init__(
            self, target_number, n_classes, patch_size=patch_size
        )

        FeatureSelectionWrapper.__init__(
            self,
            input_channels,
            sigma=sigma,
            lam=lam,
            device=device,
            target_number=target_number,
            headstart_idx=headstart_idx,
        )

    def forward(self, x):
        x = self.feature_selector.forward(x)
        x = Chen.forward(self=self, x=x)
        return x
