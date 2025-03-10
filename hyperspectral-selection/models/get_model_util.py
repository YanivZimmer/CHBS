# -*- coding: utf-8 -*-
# Torch
# utils

# from sklearn.externals
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.baseline import Baseline, BaselineFS
from models.hamida import HamidaEtAl, HamidaFS, HamidaL1
from models.chen import ChenFS,Chen
from models.extra_models import *


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    # weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    if name == "nn":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes, kwargs.setdefault("dropout", False))
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "nn_fs":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = BaselineFS(
            n_bands,
            n_classes,
            kwargs.setdefault("dropout", False),
            headstart_idx=kwargs["headstart_idx"],
        )
        lr = kwargs.setdefault("learning_rate", 0.0001)
        modified_lr = [
            {"params": list(model.parameters())[1:], "lr": lr},
            {"params": list(model.parameters())[:1], "lr": kwargs["lr_factor"] * lr},
        ]
        optimizer = optim.Adam(modified_lr, lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)

    elif name == "hamida":
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "hamida_l1":
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = True
        model = HamidaL1(n_bands, n_classes, lam=kwargs["lam"], patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "hamida_fs":
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = True
        model = HamidaFS(
            n_bands,
            n_classes,
            lam=kwargs["lam"],
            #sigma=0.2,
            #sigma=0.3,
            sigma=0.5,
            patch_size=patch_size,
            headstart_idx=kwargs["headstart_idx"],
            device=kwargs["device"],
            target_number=kwargs["bands_amount"]
        )
        lr = kwargs.setdefault("learning_rate", 0.01)
        # different learning rates ls
        #modified_lr = [
        #    {"params": list(model.parameters())[1:], "lr": lr},
        #   {"params": list(model.parameters())[:1], "lr": kwargs["lr_factor"] * lr},
        #]8
        #optimizer = optim.SGD(model.parameters(), lr=lr)#, weight_decay=0.0005)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=0.0005)
        #optimizer = optim.SGD(modified_lr, lr=lr, weight_decay=0.0005)
        #optimizer = None#LDoG(model.parameters())
        kwargs.setdefault("batch_size", 256)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "chen_fs":
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = True
        model = ChenFS(
            n_bands,
            n_classes,
            lam=kwargs["lam"],
            sigma=0.5,
            patch_size=patch_size,
            headstart_idx=kwargs["headstart_idx"],
            device=kwargs["device"],
            target_number=kwargs["bands_amount"]
        )
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # , weight_decay=0.0005)
        kwargs.setdefault("batch_size", 256)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])

    elif name == "lee":
        kwargs.setdefault("epoch", 200)
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = False
        model = LeeEtAl(n_bands, n_classes)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "lee_fs":
        kwargs.setdefault("epoch", 200)
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = False
        model = LeeEtAlFeatureSelection(n_bands, n_classes)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])

    elif name == "chen":
        patch_size = kwargs.setdefault("patch_size", 27)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.003)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 400)
        kwargs.setdefault("batch_size", 100)
    elif name == "li":
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=16, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )
        epoch = kwargs.setdefault("epoch", 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        # kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
    elif name == "hu":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = HuEtAl(n_bands, n_classes)
        # From what I infer from the paper (Eq.7 and Algorithm 1), it is standard SGD with lr = 0.01
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "he":
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault("patch_size", 7)
        kwargs.setdefault("batch_size", 40)
        lr = kwargs.setdefault("learning_rate", 0.01)
        center_pixel = True
        model = HeEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "luo":
        # All  the  experiments  are  settled  by  the  learning  rate  of  0.1,
        # the  decay  term  of  0.09  and  batch  size  of  100.
        kwargs.setdefault("patch_size", 3)
        kwargs.setdefault("batch_size", 100)
        lr = kwargs.setdefault("learning_rate", 0.1)
        center_pixel = True
        model = LuoEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.09)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "sharma":
        # We train our S-CNN from scratch using stochastic gradient descent with
        # momentum set to 0.9, weight decay of 0.0005, and with a batch size
        # of 60.  We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the validation
        # error stopped decreasing. Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault("batch_size", 60)
        epoch = kwargs.setdefault("epoch", 30)
        lr = kwargs.setdefault("lr", 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault("patch_size", 64)
        model = SharmaEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault(
            "scheduler",
            optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1
            ),
        )
    elif name == "liu":
        kwargs["supervision"] = "semi"
        # "The learning rate is set to 0.001 empirically. The number of epochs is set to be 40."
        kwargs.setdefault("epoch", 40)
        lr = kwargs.setdefault("lr", 0.001)
        center_pixel = True
        patch_size = kwargs.setdefault("patch_size", 9)
        model = LiuEtAl(n_bands, n_classes, patch_size)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # "The unsupervised cost is the squared error of the difference"
        criterion = (
            nn.CrossEntropyLoss(weight=kwargs["weights"]),
            lambda rec, data: F.mse_loss(
                rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()
            ),
        )
    elif name == "boulch":
        kwargs["supervision"] = "semi"
        kwargs.setdefault("patch_size", 1)
        kwargs.setdefault("epoch", 100)
        lr = kwargs.setdefault("lr", 0.001)
        center_pixel = True
        model = BoulchEtAl(n_bands, n_classes)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = (
            nn.CrossEntropyLoss(weight=kwargs["weights"]),
            lambda rec, data: F.mse_loss(rec, data.squeeze()),
        )
    elif name == "mou":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        kwargs.setdefault("epoch", 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault("lr", 1.0)
        model = MouEtAl(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=epoch // 4, verbose=True
        ),
    )
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault("batch_size", 100)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs
#