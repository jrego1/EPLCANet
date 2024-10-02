import os
from random import randint

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
import h5py
from imageio import imwrite
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import pickle
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize, RandomResizedCrop

from lcapt.analysis import make_feature_grid
from lcapt.lca import LCAConv2D
from lcapt.preproc import make_unit_var, make_zero_mean

from data_utils import *
from model_utils_fast import make_pools

EPOCHS = 35
mbs = 128
data_aug = True

lca_lambda = 0.25
n_feats = 64
ksize = 5
lca_iters = 600
dict_learning = 'builtin'
dict_loss = 'combo'

pools = make_pools('immm')

lca_load_dir = f'/storage/jr3548@drexel.edu/LCANet/pretrained/dictionary_learning/'
lca_result_dir = f'/storage/jr3548@drexel.edu/LCANet/{dict_learning}/{dict_loss}/{lca_lambda}/'

if not os.path.isdir(lca_result_dir):
    print("Creating directory :", lca_result_dir)
    os.makedirs(lca_result_dir)

def hard_sigmoid(x):
    return (1 + F.hardtanh(2 * x - 1)) * 0.5

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu()
    return hook

def get_layer_names(model):
    layer_names = []
    for name, _ in model.named_parameters():
        if ('conv' in name) or ('fc' in name):
            layer_names.append('.'.join(name.split('.')[:-1]))
    return list(set(layer_names))

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
    
def accuracy(y_hat, y):
    return (y_hat.argmax(-1) == y).float().mean().item()

def train_epoch(train_loader, model, optimizer, loss_fn, scheduler):
    avg_acc = []
    avg_loss = []
    model.train()
    
    for x, y in train_loader:
        x = make_unit_var(make_zero_mean(x))
        y_hat, lca_acts, recon_errors = model(x.to(device=1))
        
        avg_acc.append(accuracy(y_hat, y.to(y_hat.device.index)))
        loss = loss_fn(y_hat, y.to(y_hat.device.index)) # Supervised loss
        avg_loss.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if dict_learning == 'builtin' and (dict_loss == 'combo' or dict_loss == 'recon'): # Update LCA weights again based on LCA  ()
            lca.update_weights(lca_acts, recon_errors) # Unsupervised loss

    lca_sparsity = (lca_acts != 0).float().mean().item()
    
    return model, sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss), lca_sparsity, recon_errors.mean().item()

def adv_train_epoch(train_loader, model, optimizer, loss_fn, scheduler):
    avg_acc = []
    avg_loss = []
    model = PyTorchClassifier(
        model,
        loss_fn,
        (3, 40, 40),
        10,
        optimizer,
        clip_values=(0, 1),
    )
    attack = ProjectedGradientDescentPyTorch(
        model,
        2,
        0.75,
        2.5 * 0.75 / 20,
        max_iter=20,
        batch_size=train_loader.batch_size,
        verbose=False
    )
    for x, y in train_loader:
        y = y.cuda()
        model.eval()
        x_adv = attack.generate(x.cpu().numpy())
        model.train()
        y_hat, lca_acts, recon_errors = model.model(torch.from_numpy(x_adv))
        avg_acc.append(accuracy(y_hat, y))
        loss = loss_fn(y_hat, y)
        avg_loss.append(loss.item())
        model.model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    lca_sparsity = (lca_acts != 0).float().mean().item()
    return model.model, sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss), lca_sparsity, recon_errors.mean().item() 

def val_epoch(val_loader, model, loss_fn):
    avg_acc = []
    avg_loss = []
    model.eval()
    for x, y in val_loader:
        y_hat, lca_acts, recon_errors = model(x.to(device=1))
        avg_acc.append(accuracy(y_hat, y.to(y_hat.device.index)))
        loss = loss_fn(y_hat, y.to(y_hat.device.index))
        avg_loss.append(loss.item())

    lca_sparsity = (lca_acts != 0).float().mean().item()
    return sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss),  lca_sparsity, recon_errors.mean().item()

def train(model, train_loader, val_loader, optimizer, loss_fn, scheduler, n_epochs, adv_training=False):
    print('Training...')
    train_acc_list, train_loss_list, train_recon_error_list, val_acc_list, val_loss_list, val_recon_error_list, train_sparsity_list, val_sparsity_list, lr_list = ([] for _ in range(9))
    for epoch in range(1, n_epochs + 1):
        if adv_training:
            model, train_acc, train_loss, train_sparsity, train_recon_error = adv_train_epoch(train_loader, model, optimizer, loss_fn, scheduler)
            val_acc, val_loss, val_sparsity, val_recon_error = val_epoch(val_loader, model, loss_fn)
        else:
            model, train_acc, train_loss, train_sparsity, train_recon_error = train_epoch(train_loader, model, optimizer, loss_fn, scheduler)
            val_acc, val_loss, val_sparsity, val_recon_error = val_epoch(val_loader, model, loss_fn)
        print(
            f'Train Acc: {round(train_acc, 2)}; ',
            f'Train Loss: {round(train_loss, 4)}; ',
            f'Val Acc: {round(val_acc, 2)}; ',
            f'Val Loss: {round(val_loss, 4)} ',
            f'Train Sparsity: {round(train_sparsity, 4)}; ',
            f'Val Sparsity: {round(val_sparsity, 4)}; ',
            f'Train Recon Error: {round(train_recon_error, 2)}',
            f'Val Recon Error: {round(val_recon_error, 2)}'
            f'LR: {scheduler.get_last_lr()[0]:.2e}'
        )
        
        train_acc_list.append(round(train_acc, 2))
        train_loss_list.append(round(train_loss, 4))
        train_recon_error_list.append(round(train_recon_error, 4))
        val_acc_list.append(round(val_acc, 2))
        val_loss_list.append(round(val_loss, 4))
        val_recon_error_list.append(round(val_recon_error, 4))
        train_sparsity_list.append(round(train_sparsity, 2))
        val_sparsity_list.append(round(val_sparsity , 2))
        lr_list.append(scheduler.get_last_lr()[0])

        if epoch == n_epochs - 1:
            torch.save(model, os.path.join(lca_result_dir, 'model.pt'))
    
    df = pd.DataFrame({
        'Train Accuracy': train_acc_list,
        'Train Loss': train_loss_list,
        'Train Reconstruction Error': train_recon_error_list,
        'Validation Accuracy': val_acc_list,
        'Validation Loss': val_loss_list,
        'Val Reconstruction Error': val_recon_error_list,
        'Train Sparsity': train_sparsity_list,
        'Validation Sparsity': val_sparsity_list,
        'Learning Rate': lr_list
    })
    
    
    return model, df

if data_aug:
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
else:
    transform_train = torchvision.transforms.Compose(
        [
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
    download=True
)
cifar10_test_dset = torchvision.datasets.CIFAR10(
    "/storage/jr3548@drexel.edu/eplcanet/data/cifar10_pytorch",
    train=False,
    transform=transform_test,
    download=True
)

val_index = np.random.randint(10)
val_samples = list(range(5000 * val_index, 5000 * (val_index + 1)))

train_loader = torch.utils.data.DataLoader(
    cifar10_train_dset, batch_size=mbs, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    cifar10_test_dset, batch_size=200, shuffle=False, num_workers=2, pin_memory=True 
)

lca = LCAConv2D(
    n_feats,
    3,
    lca_load_dir if dict_learning else lca_result_dir,
    ksize,
    2,
    lca_lambda,
    100,
    5e-2,
    lca_iters,
    pad = "same",
    return_vars=['inputs', 'acts', 'recons', 'recon_errors','states'],
    req_grad=False if dict_loss == 'recon' else True,
    input_zero_mean=False,
    input_unit_var=False,
    nonneg=True
)

if dict_learning == 'train':
    lca = nn.DataParallel(lca, [0, 1]).cuda()
elif dict_learning == 'pretrained':
    ckpt = torch.load(os.path.join(lca_load_dir, 'model.pt'), map_location='cpu')
    lca.assign_weight_values(ckpt.weights)
    
if dict_learning == 'train': # Only train dictionary, apart from CNN
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch}')
        if (epoch + 1) % (EPOCHS // 7) == 0:
            lca.module.lambda_ += 0.1
        for x, _ in train_loader:
            x = make_unit_var(make_zero_mean(x))
            acts, recon_errors = lca(x)
            lca.module.update_weights(acts, recon_errors)
        torch.save(lca.module, os.path.join(lca_load_dir, 'model.pt'))

class LCANet(nn.Module):
    def __init__(self, lca):
        super(LCANet, self).__init__()
        
        self.lca = lca
        self.conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False) # Originally true, maybe that helped me?
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.fc = nn.Linear(512, 10)  # 512 channels, 3x3 spatial size, 10 output classes... shouldnt it be 2048?
        
        self.pool = nn.MaxPool2d(2, 2)
        self.pool.return_indices = True
    
    def forward(self, x):
        inputs, lca_acts, recons, recon_errors, states  = self.lca(x)

        x = lca_acts # LCA layer
        x = F.relu(self.pool(self.conv1(x))[0])
        x = F.relu(self.pool(self.conv2(x))[0])
        x = F.relu(self.pool(self.conv3(x))[0]) 
        x = F.relu(self.pool(self.conv4(x))[0])
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x, lca_acts, recon_errors


model = LCANet(lca)
model.to(device=1)
print(model)

optimizer = torch.optim.Adam(model.parameters(), 1e-3)
loss_fn = nn.CrossEntropyLoss()
scheduler = OneCycleLR(
    optimizer=optimizer,
    max_lr=0.01,
    epochs=EPOCHS,
    steps_per_epoch=len(cifar10_train_dset) // mbs,
    div_factor=1e2,
    final_div_factor=1e3,
    three_phase=True,
    total_steps=len(train_loader) * EPOCHS
)

model, df = train(model, train_loader, test_loader, optimizer, loss_fn, scheduler, EPOCHS)

df.to_csv(lca_result_dir + 'results.csv')

torch.save(model, os.path.join(lca_result_dir, 'model.pt'))

plot_metrics(df['Train Accuracy'], df['Validation Accuracy'], 'Accuracy', lca_result_dir + 'accuracy.png')
plot_metrics(df['Train Loss'], df['Validation Loss'], 'Loss', lca_result_dir + 'loss.png')
plot_metrics(df['Train Sparsity'], df['Validation Sparsity'], 'LCA Activation Sparsity', lca_result_dir + 'sparsity.png')
plot_metrics(df['Train Reconstruction Error'], df['Validation Reconstruction Error'], 'Reconstruction Error', lca_result_dir + 'recon_error.png')