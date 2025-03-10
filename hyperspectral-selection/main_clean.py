# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""

# Python 2/3 compatiblity
from __future__ import division, print_function
import os

from common_utils.main_utils import Trainer

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["WORLD_SIZE"] = "1"
import torch

#
#import os
import argparse
import json

# Numpy, scipy, scikit-image, spectral
import numpy as np
# Visualization
import seaborn as sns
import sklearn.svm
# Torch
import torch
torch.set_float32_matmul_precision('medium')
import torch.utils.data as data
import visdom
from torchsummary import summary

from common_utils.kfold import CrossValidator
from datasets_utils.datasets import DATASETS_CONFIG, HyperX, get_dataset, open_file
from train_test.models_train_test import save_model, test, train
from models.get_model_util import get_model
from common_utils.utils import (build_dataset, compute_imf_weights, convert_from_color_,
                                convert_to_color_, display_dataset, display_predictions,
                                explore_spectrums, get_device, metrics, plot_spectrums,
                                sample_gt, show_results)

dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default=True, choices=dataset_names, help="To use stg"
)
parser.add_argument(
    "--stg", type=str, default=True, help="Dataset to use."
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model to train. Available:\n"
    "SVM (linear), "
    "SVM_grid (grid search on linear, poly and RBF kernels), "
    "baseline (fully connected NN), "
    "hu (1D CNN), "
    "hamida (3D CNN + 1D classifier), "
    "lee (3D FCN), "
    "chen (3D CNN), "
    "li (3D CNN), "
    "he (3D CNN), "
    "luo (3D CNN), "
    "sharma (2D CNN), "
    "boulch (1D semi-supervised CNN), "
    "liu (3D semi-supervised CNN), "
    "mou (1D RNN)",
)

parser.add_argument(
    "--folder",
    type=str,
    help="Folder where to store the "
    "datasets (defaults to the current working directory).",
    default="./Datasets/",
)
parser.add_argument(
    "--cuda",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument(
    "--restore",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)

# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=0.1,
    help="Percentage of samples to use for training (default: 10%)",
)
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode" " (random sampling or disjoint, default: random)",
    default="random",
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default=None,
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default=None,
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)
# Training options
group_train = parser.add_argument_group("Training")
group_train.add_argument(
    "--epoch",
    type=int,
    help="Training epochs (optional, if" " absent will be set by the model)",
)
group_train.add_argument(
    "--epoch_second", type=int, help="Training epochs after feature selection"
)

group_train.add_argument(
    "--lam", type=float, help="lam for regularization in feature selection"
)
group_train.add_argument("--lr_factor", type=int, help="multiply lr by it")


group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--lr", type=float, help="Learning rate, set by the model if not specified."
)
group_train.add_argument(
    "--class_balancing",
    action="store_true",
    help="Inverse median frequency class balancing (default = False)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)

group_train.add_argument(
    "--bands_amount",
    type=int,
    help="Bands amount",
)


group_train.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument(
    "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
)
group_da.add_argument(
    "--radiation_augmentation",
    action="store_true",
    help="Random radiation noise (illumination)",
)
group_da.add_argument(
    "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
)

parser.add_argument(
    "--with_exploration", action="store_true", help="See data exploration visualization"
)
parser.add_argument(
    "--download",
    type=str,
    default=None,
    nargs="+",
    choices=dataset_names,
    help="Download the specified datasets and quits.",
)


args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Number of epochs to run
EPOCH2 = args.epoch_second
STG_USE=args.stg
# lam for regularization
LAM = args.lam

# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride
BANDS_AMOUNT = args.bands_amount

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + " " + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


def read_dict(filename):
    with open(filename) as f:
        data = f.read()
    print("Data type before reconstruction : ", type(data))
    # reconstructing the data as a dictionary
    js = json.loads(data)
    return js

# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
# Number of classes
# N_CLASSES = len(LABEL_VALUES) -  len(IGNORED_LABELS)
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]


if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}

'''
def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)
'''
def get_hyperparams():
    hyperparams = vars(args)

    # Instantiate the experiment based on predefined networks
    hyperparams.update(
        {
            "n_classes": N_CLASSES,
            "n_bands": N_BANDS,
            "ignored_labels": IGNORED_LABELS,
            "device": CUDA_DEVICE,
        }
    )
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    return hyperparams

def model_creator_func(**hyperparams):
    # model, optimizer, loss, hyperparams
    return get_model(MODEL, **hyperparams)

def add_or_append(key,value,res_dict):
    if key not in res_dict:
        res_dict[key] = [value]
    else:
        res_dict[key].append(value)
def nice_print(res_dict):
    print("res_dict")
    for key,value in res_dict.items():
        print(key,"=",value)

if __name__ == '__main__':
    #lm = LAM
    hyperparams = get_hyperparams()
    results = []
    all_algo_n_bands_to_selection = read_dict(f'algo_bands_mapping_results_temp_{DATASET}.json')
    train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    train_gt, val_gt = sample_gt(train_gt, 0.999, mode="random")
    # Generate the dataset
    hyperparams["headstart_idx"] = None  # if use_stg else n_bands_to_selection[str(n_selected_bands)]
    hyperparams["lam"] = LAM
    #cctor mess with hyperparams
    #hyperparams1=dict(hyperparams)
    trainer = Trainer(model_creator=lambda: model_creator_func(**hyperparams),mode_name=MODEL, img=img, gt=train_gt,
                      display=viz,device=CUDA_DEVICE,hyperparams1=hyperparams)
    losses = {}
    regs = {}
    accs = {}
    opt_to_gates = {}
    #
    x = 0.5
    y = 2.1
    step = 0.2
    opt_with_lr_to_bands_acc={}
    for lr in reversed([0.001]):
        nice_print(opt_with_lr_to_bands_acc)
        for lm in np.arange(x, y + step, step):
            for opt in ['ADAM', 'DOG']:#,'DOG5','DOG']:#,'DOG8','DOG9','DOG10','DOG5','DOG4','SGD','ADAM']:
                opt_with_lr_lam = f'{opt}_{lr}_{lm}'
                short_key = f'{opt}_{lr}'
                hyperparams["lam"] = lm
                model, _, loss, hyperparams = get_model(MODEL, **hyperparams)
                losses[opt_with_lr_lam], regs[opt_with_lr_lam] = trainer.train(net=model, optimizer_name=opt, criterion=loss, epoch=EPOCH,
                              lam=lm, lr=lr, display_iter=1000, device=torch.device('cuda'), display=viz)
                acc, gates, gates_prob_one, gates_positive_prob = trainer.test(net=model)
                accs[opt_with_lr_lam] = acc
                opt_to_gates[opt_with_lr_lam] = gates_prob_one
                if short_key not in opt_with_lr_to_bands_acc:
                    opt_with_lr_to_bands_acc[short_key]={}
                add_or_append(gates_positive_prob,acc,opt_with_lr_to_bands_acc[short_key])
                print(opt,lm,lr, "prob1", gates_prob_one, "prob_pos", gates_positive_prob, acc)
    #print("losses", losses)
    #print("regs", regs)
    # #print("acc", acc)
    nice_print(opt_with_lr_to_bands_acc)
    trainer.draw_optimizer(losses, regs, accs, lm, 999)
