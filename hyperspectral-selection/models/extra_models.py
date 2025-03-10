import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from tqdm import tqdm

from feature_selector.feature_selector_general import FeatureSelector


class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0
        )

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        return x


class LeeEtAlFeatureSelection(LeeEtAl):
    def __init__(self, input_channels, n_classes):
        super(LeeEtAlFeatureSelection, self).__init__(input_channels, n_classes)
        self.input_channels = input_channels
        self.feature_selector = FeatureSelector(
            self.input_channels, sigma=0.25, device="cuda:0"
        )
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.reg = self.feature_selector.regularizer
        self.lam = 1.25  # 0.75
        self.mu = self.feature_selector.mu
        self.sigma = self.feature_selector.sigma
        self.test = False
        self.k = int(0.4 * input_channels)
        self.mask = None

    # def set_fs_device(self,device):
    #    self.FeatureSelector = FeatureSelector(self.input_channels, sigma=0.5, device=device)

    def forward(self, x):
        if self.test and self.feature_selector.mask is None:
            self.feature_selector.set_mask(self.get_top_k_gates(self.k))
        x = self.feature_selector.forward(x)
        x = LeeEtAl.forward(self=self, x=x)
        return x

    def regularization(self):
        reg = torch.mean(self.reg((self.mu + 0.5) / self.sigma))
        return self.lam * reg

    def get_gates(self, mode):
        if mode == "raw":
            return self.feature_selector.mu.detach().cpu().numpy()
        elif mode == "prob":
            return np.minimum(
                1.0,
                np.maximum(0.0, self.feature_selector.mu.detach().cpu().numpy() + 0.5),
            )
        else:
            raise NotImplementedError()

    def get_top_k_gates(self, k):
        gates = self.get_gates("prob")
        k_max_val = torch.topk(torch.from_numpy(gates), k).values[k - 1]
        mask = torch.from_numpy(gates) > k_max_val
        return mask




class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes, (3, 3, 3), padding=(1, 0, 0))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1, 2, 2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1, 2, 2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1, 1, 1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = self.pool1(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = self.pool2(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv3(x))
            print(x.size())
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LiuEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(LiuEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(self.features_sizes[2], self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2], self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1], self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0], input_channels)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        x = self.conv1_bn(x)
        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        x = x.squeeze()
        x_conv1 = self.conv1_bn(self.conv1(x))
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).view(-1, self.features_sizes[2])
        x = x_enc

        x_classif = self.fc_enc(x)

        # x = F.relu(self.fc1_dec_bn(self.fc1_dec(x) + x_enc))
        x = F.relu(self.fc1_dec(x))
        x = F.relu(
            self.fc2_dec_bn(self.fc2_dec(x) + x_pool1.view(-1, self.features_sizes[1]))
        )
        x = F.relu(
            self.fc3_dec_bn(self.fc3_dec(x) + x_conv1.view(-1, self.features_sizes[0]))
        )
        x = self.fc4_dec(x)
        return x_classif, x


class BoulchEtAl(nn.Module):
    """
    Autoencodeurs pour la visualisation d'images hyperspectrales
    A.Boulch, N. Audebert, D. Dubucq
    GRETSI 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, planes=16):
        super(BoulchEtAl, self).__init__()
        self.input_channels = input_channels
        self.aux_loss_weight = 0.1

        encoder_modules = []
        n = input_channels
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            print(x.size())
            while n > 1:
                print("---------- {} ---------".format(n))
                if n == input_channels:
                    p1, p2 = 1, 2 * planes
                elif n == input_channels // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, 3, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()

        self.classifier = nn.Linear(self.features_sizes, n_classes)
        self.regressor = nn.Linear(self.features_sizes, input_channels)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c * w

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(-1, self.features_sizes)
        x_classif = self.classifier(x)
        x = self.regressor(x)
        return x_classif, x


class MouEtAl(nn.Module):
    """
    Deep recurrent neural networks for hyperspectral image classification
    Lichao Mou, Pedram Ghamisi, Xiao Xang Zhu
    https://ieeexplore.ieee.org/document/7914752/
    """

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data, -0.1, 0.1)
            init.uniform_(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(MouEtAl, self).__init__()
        self.input_channels = input_channels
        self.gru = nn.GRU(1, 64, 1, bidirectional=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(64 * input_channels)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(64 * input_channels, n_classes)

    def forward(self, x):
        x = x.squeeze()
        x = x.unsqueeze(0)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        x = self.gru(x)[0]
        # x is in C, N, 64, we permute back
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        x = self.gru_bn(x)
        x = self.tanh(x)
        x = self.fc(x)
        return x

