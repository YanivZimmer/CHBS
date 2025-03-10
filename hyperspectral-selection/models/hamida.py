import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from feature_selector.feature_selector_general import FeatureSelector
from feature_selector.feature_selector_l1_conv import FeatureSelectorL1Conv
from .feature_selector_wrapper import FeatureSelectionWrapper
from .feature_selector_l1_wrapper import FeatureSelectionL1Wrapper
from feature_selector.concrete_autoencoder import ConcreteEncoder

PATH="hamida_weights1"

class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self):
        pass
    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)
        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0
            )
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        # self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class HamidaFS(HamidaEtAl, FeatureSelectionWrapper):
    def __init__(
        self,
        input_channels,
        n_classes,
        patch_size=5,
        dilation=1,
        lam=1,
        sigma=0.5,
        headstart_idx=None,
        device="cuda:0",
        target_number=10        
    ):
        HamidaEtAl.__init__(
            self, target_number, n_classes, patch_size=patch_size, dilation=dilation
        )
        self.downstream_model = HamidaEtAl
        FeatureSelectionWrapper.__init__(
            self,
            input_channels,
            sigma=sigma,
            lam=lam,
            device=device,
            target_number=target_number,
            headstart_idx=headstart_idx,
        )
        #for param in HamidaEtAl.parameters(self=HamidaEtAl):
        #  param.requires_grad = False
        self.fs_params=[]
        for module in self.modules():
          if isinstance(module, ConcreteEncoder) or isinstance(module, FeatureSelector):
            for param in module.parameters():
              print("param here is shape",param.shape)
              self.fs_params.append(param)
        print("all fs.params:")
        for i,elm in enumerate(self.fs_params):
            print("elm",i,elm.shape)
        #for name, param in self.named_parameters():
        #  if "HamidaEtAl" in name:
        #    param.requires_grad = False
        #HamidaEtAl.load_state_dict(torch.load(PATH))
        #HamidaEtAl.load_state_dict(torch.load(CHECKPOINT))
    def forward(self, x):
        x = self.feature_selector.forward(x)
        #with torch.no_grad():
        x = HamidaEtAl.forward(self=self, x=x)
        return x


class HamidaFS123(nn.Module):#HamidaEtAl, FeatureSelectionWrapper):
    def __init__(
        self,
        input_channels,
        n_classes,
        patch_size=5,
        dilation=1,
        lam=1,
        sigma=0.5,
        headstart_idx=None,
        device="cuda:0",
        target_number=10        
    ):
        self.down_model=HamidaEtAl(target_number, n_classes, patch_size=patch_size, dilation=dilation)
        self.feature_selector_wrapper=\
        FeatureSelectionWrapper(
            input_channels,
            sigma=sigma,
            lam=lam,
            device=device,
            target_number=target_number,
            headstart_idx=headstart_idx,
        )

    def forward(self, x):
        x = self.feature_selector_wrapper.feature_selector.forward(x)
        x = self.down_model.forward(self=self, x=x)
        return x


class HamidaL1(HamidaEtAl, FeatureSelectionL1Wrapper):
    def __init__(
        self,
        input_channels,
        n_classes,
        lam,
        patch_size=5,
        dilation=1,
        device="cuda:0",
    ):
        HamidaEtAl.__init__(
            self, input_channels, n_classes, patch_size=patch_size, dilation=dilation
        )
        FeatureSelectionL1Wrapper.__init__(self, input_channels,lam)

    def forward(self, x):
        x = FeatureSelectionL1Wrapper.forward(self=self, x=x)
        x = HamidaEtAl.forward(self=self, x=x)
        return x
