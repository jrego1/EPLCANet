import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

import os
from datetime import datetime
import time
import math
import pandas as pd
from data_utils import *
from pathlib import Path
from glob import glob
from itertools import repeat
from torch.nn.parameter import Parameter
import collections
import matplotlib
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize, RandomResizedCrop, RandomCrop, Normalize
# from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
# from art.attacks.evasion.hop_skip_jump import HopSkipJump
# from art.estimators.classification import PyTorchClassifier

from lcapt.analysis import make_feature_grid
from lcapt.preproc import make_zero_mean, make_unit_var
from lcapt.lca import LCAConv2D
from lcapt.metric import compute_l1_sparsity, compute_l2_error, compute_times_active_by_feature



EPOCHS = 35
DEVICE = 1
subset_train = 0.5


FEATURES = 64  # number of dictionary features to learn
KERNEL_SIZE = 9  # height and width of each feature
LAMBDA = 0.2  # LCA threshold
LCA_ITERS = 600 # LCA iterations to run before adding feedback, reduced later
LEARNING_RATE = 1e-2
TAU = 100
STRIDE = 2

PATH = '/storage/jr3548@drexel.edu/eplcanet/results/CIFAR10/'

class InputNorm(nn.Module):
    def __init__(self):
        super(InputNorm, self).__init__()

    def forward(self, x):
        return make_unit_var(make_zero_mean(x))

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class InputNorm(nn.Module):
    def __init__(self):
        super(InputNorm, self).__init__()

    def forward(self, x):
        return make_unit_var(make_zero_mean(x))

lca = LCAConv2D(
            out_neurons=FEATURES,
            in_neurons=3,
            result_dir= PATH + 'subset_dictlearning/',
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            lambda_=LAMBDA,
            tau=TAU,
            eta=LEARNING_RATE,
            lca_iters = LCA_ITERS,
            input_zero_mean=False,
            input_unit_var=False,
            return_vars=['inputs', 'acts', 'recons', 'recon_errors','states'],
        )

lca = lca.to(DEVICE)

if __name__ == "__main__":
    #if args.data_aug:
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomCrop(
                size=[32, 32], padding=4, padding_mode="edge"
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010),
            ),
        ]
    )
    
    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010),
            ),
        ]
        )

    cifar10_train_dset = torchvision.datasets.CIFAR10(
        "/storage/jr3548@drexel.edu/eplcanet/data/cifar10_pytorch",
        train=True,
        transform=transform_train,
        download=True,
    )

    # train dictionary with a subset of data, we also need data to train EP and fine tune the dict, save indices to exclude from EP training
    subset_size = int(subset_train * len(cifar10_train_dset))
    random_indices = np.random.permutation(len(cifar10_train_dset))[:subset_size]
    np.save(PATH +'subset_dictlearning/train_indices.npy', random_indices)

    # train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=mbs, sampler = torch.utils.data.SubsetRandomSampler(val_samples), shuffle=False, num_workers=1)
    train_loader = torch.utils.data.DataLoader(
        cifar10_train_dset, batch_size=200, shuffle=True, num_workers=2, pin_memory=True, drop_last=True

    )

for epoch in range(EPOCHS):
    print(f'Epoch: {epoch}')
    if (epoch + 1) % (EPOCHS // 7) == 0:
        lca.lambda_ += 0.1

    for x, _ in train_loader:
        x = make_unit_var(make_zero_mean(x))
        inputs, acts, recons, recon_errors, states = lca(x.to(DEVICE))
        lca.update_weights(acts, recon_errors)
    torch.save(lca, os.path.join(PATH + 'subset_dictlearning/', 'dictionary.pt'))
    
#plot_activities(lca.module, PATH + "lca_activations.png")
plot_lca_weights(lca, PATH + "lca_weights.png")