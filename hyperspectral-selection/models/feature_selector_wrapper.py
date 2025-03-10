import numpy as np
import torch

from feature_selector.feature_selector_general import FeatureSelector
from feature_selector.feature_selector_mega import FeatureSelectorMega
from feature_selector.concrete_autoencoder import ConcreteEncoder


class FeatureSelectionWrapper:
    def __init__(
        self,
    ):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def __init__(
        self,
        input_channels,
        sigma=0.5,
        lam=1,
        device="cuda:0",
        target_number=None,
        headstart_idx=None,
    ):
        self.input_channels = input_channels
        self.feature_selector = FeatureSelector(
             self.input_channels, sigma=sigma, device=device, target_number=target_number, headstart_idx=headstart_idx
        )
        # # self.feature_selector = FeatureSelectorMega(
        #      self.input_channels, device=device, target_number=target_number
        # )
        #self.feature_selector = ConcreteEncoder(input_dim=self.input_channels, output_dim=target_number, device=device,headstart_idx=headstart_idx)
        self.target_number = target_number
        self.test = False
        self.k = None
        self.reg = self.feature_selector.regularizer if hasattr(self.feature_selector,"regularizer") else lambda x:0 #
        self.sigma = sigma
        self.lam = lam
        self.device = device
        self.headstart_idx = headstart_idx
        self.mu = self.feature_selector.mu if hasattr(self.feature_selector,"mu") else None

    def reset_gates(self):
        print("The device is ", self.device)
        # self.feature_selector = FeatureSelector(
        #     self.input_channels, sigma=self.sigma, device=self.device, target_number=self.target_number, headstart_idx=None#self.headstart_idx
        # )
        # self.mu = self.feature_selector.mu
        self.feature_selector = ConcreteEncoder(input_dim=self.input_channels, output_dim=self.target_number, device=self.device)


    def forward(self, x):
        if self.test and self.feature_selector.mask is None:
            self.feature_selector.set_mask(self.get_top_k_gates(self.k))
        return self.feature_selector.forward(x)

    def regularization(self):
        if self.mu is None:
            return 0
        # If fs is using constant masking regularization of it should be 0
        if self.feature_selector.mask is not None:
            return torch.Tensor([0]).to(self.feature_selector.device)
        reg = torch.mean(self.reg((self.mu + 0.5) / self.sigma))
        total_reg = self.lam * reg
        return total_reg

    def get_gates(self, mode):
        return self.feature_selector.get_gates(mode)

    def get_top_k_gates(self, k):
        gates = self.get_gates("prob")
        k_max_val = torch.topk(torch.from_numpy(gates), k).values[k - 1]
        mask = torch.from_numpy(gates) >= k_max_val
        return mask

    def update_bands(self, bands):
        shifted_vector = np.array(bands) - 1
        tensor = torch.zeros(self.input_channels, dtype=torch.bool)
        tensor[shifted_vector] = True
        self.feature_selector.mask = tensor
        # return tensor
