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
matplotlib.use('Agg')
characteristic_time = np.zeros((250,2))
characteristic_param = 0
attack_param = 0
# Activation functions
def my_sigmoid(x):
    return 1/(1+torch.exp(-4*(x-0.5)))

def hard_sigmoid(x):
    return (1+F.hardtanh(2*x-1))*0.5

def ctrd_hard_sig(x):
    return (F.hardtanh(2*x))*0.5

def my_hard_sig(x):
    return (1+F.hardtanh(x-1))*0.5


# Some helper functions
def grad_or_zero(x):
    if x.grad is None:
        return torch.zeros_like(x).to(x.device)
    else:
        return x.grad

def neurons_zero_grad(neurons):
    for idx in range(len(neurons)):
        if neurons[idx].grad is not None:
            neurons[idx].grad.zero_()

def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy

class InputNorm(nn.Module):
    def __init__(self):
        super(InputNorm, self).__init__()

    def forward(self, x):
        return make_unit_var(make_zero_mean(x))

class Identity2d(nn.Module):
    # Added to return indices, allow for identity to work with maxpool layers in same network
    def __init__(self, return_indices):
        super(Identity2d, self).__init__()
        self.kernel_size = 1
        self.stride = 1
        self.return_indices = return_indices

    def forward(self, x, indices=None):
        # Generate dummy indices that match the shape of the input
        x = torch.nn.Identity()(x)
        indices = torch.arange(len(x))
        if self.return_indices:
            return x, indices
        else:
            return x

def make_pools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p]=='m':
            pools.append( torch.nn.MaxPool2d(2, stride=2,return_indices=True) )
        elif letters[p]=='a':
            pools.append( torch.nn.AvgPool2d(2, stride=2) )
        elif letters[p]=='i':
            pools.append( Identity2d(return_indices=True))
    return pools



def make_unpools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p]=='m':
            pools.append( torch.nn.MaxUnpool2d(2, stride=2))
        elif letters[p]=='a':
            pools.append( torch.nn.AvgPool2d(2, stride=2) )
        elif letters[p]=='i':
            pools.append( Identity2d(return_indices=False) )      
    return pools


       
def my_init(scale): # Experiment with this?
    def my_scaled_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)
    return my_scaled_init
    
# Normal Convolutional Neural Network

class P_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, unpools, paddings, activation=hard_sigmoid, softmax=False):
        super(P_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.fc = fc
        self.nc = fc[-1]        

        self.activation = activation
        self.pools = pools
        self.unpools = unpools
        self.poolidxs = []
        self.synapses = torch.nn.ModuleList()
        
        self.softmax = softmax # whether to use softmax readout or not

        size = in_size # size of the input : 32 for cifar10

        for idx in range(len(channels)-1): 
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
                                                 stride=strides[idx], padding=paddings[idx], bias=True))
                
            size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool

        size = size * size * channels[-1]        
        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
            
    def init_poolidxs(self, mbs, device):
        
        self.poolidxs = []
        append = self.poolidxs.append
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )   # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size),  device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]),  device=device))
        else:
            # we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), device=device))            
          
        return
    
    def init_neurons(self, x):
        
        mbs=x.size(0)
        device=x.device
        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )   # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True,  device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True,  device=device))
        else:
            # we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            
          
        return neurons

    def Phi(self, x, y, neurons, beta, criterion):
        # Not used in faster forward function, but left here for reference, used in original Laborieux code for energy comp
        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons
        phi = 0 #torch.zeros(x.shape[0], device=x.device, requires_grad=True)

        #Phi computation changes depending on softmax == True or not
        if not self.softmax:
            for idx in range(conv_len):    
                phi = phi + torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
            for idx in range(conv_len, tot_len):
                phi = phi + torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].float(), y).squeeze()             
                phi = phi - beta*L

        else:
            # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            for idx in range(conv_len):
                phi = phi + torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
            for idx in range(conv_len, tot_len-1):
                phi = phi + torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                phi = phi - beta*L            
        
        return phi
    

    def forward(self, x, y=0, neurons=None, T=29, beta=0.0, scale_feedback=1.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        global characteristic_param, characteristic_time, attack_param
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)       
        conv_len = len(self.synapses) - 1 #5
        tot_len = len(self.synapses) #6
   
        self.poolsidx = self.init_poolidxs(mbs,x.device)
        unpools = make_unpools('mmmm')
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = True
        
        layers = [x] + neurons
        new_layers = [] # tendency of neurons
        
        for neuron in neurons: # exclude input layer
            new_layers.append(torch.zeros_like(neuron, device=x.device))
            
        for t in range(T):
            cost = 0
            for idx in range(conv_len):
                new_layers[idx],self.poolidxs[idx] = self.pools[idx](self.synapses[idx](layers[idx])) 
            for idx in range(conv_len-1): 
                new_layers[idx] = new_layers[idx] + F.conv_transpose2d(unpools[idx](layers[idx+2],self.poolidxs[idx+1]),self.synapses[idx+1].weight,padding=self.paddings[idx+1])
            if(tot_len-conv_len!=1):
                new_layers[conv_len-1] = new_layers[conv_len-1] + torch.matmul(layers[conv_len+1],self.synapses[conv_len].weight).reshape(new_layers[conv_len-1].shape)
                    
            
            if self.softmax:
                # FC layer not in neurons
                for idx in range(conv_len, tot_len-1): #-1 5, 5
                    new_layers[idx] = self.synapses[idx](layers[idx].view(mbs,-1)) #+ torch.matmul(layers[idx+1],self.synapses[idx+1].weight.T)
                for idx in range(conv_len, tot_len-2): # 5, 4 ?
                    new_layers[idx] = new_layers[idx] + torch.matmul(layers[idx+2],self.synapses[idx+1].weight)
                    
                if beta!=0.0:
                    y_hat = F.softmax(self.synapses[-1](layers[-1].view(x.size(0),-1)), dim = 1)  # Apply fc layer
                    cost = beta*torch.matmul((F.one_hot(y, num_classes=self.nc)-y_hat),self.synapses[-1].weight) # Compute cost from output
                    cost = cost.reshape(layers[-1].shape)
                    

            for idx in range(tot_len-1): # 5
                if idx==tot_len-2 and beta!=0: # 4
                    layers[idx+1] = self.activation(new_layers[idx]+cost).detach()
                else:
                    layers[idx+1] = self.activation(new_layers[idx]).detach()
                
                layers[idx+1].requires_grad = True
        
        return layers[1:],self.poolidxs
    
    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = False
        self.zero_grad()            # p.grad is zero
        
        # LCA layers included in phi computation?
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem

    def compute_syn_grads_alternate(self, x, y, neurons_1, neurons_2,pools_idx_1, pools_idx_2, betas, criterion, check_thm=False):
        neurons_1 = [x] + neurons_1
        neurons_2 = [x] + neurons_2
        
        
        
        for i in range(len(self.kernels)):
            temp1 = F.conv2d(neurons_1[i].permute(1,0,2,3),self.unpools[0](neurons_1[i+1], pools_idx_1[i]).permute(1,0,2,3),stride=1,padding=self.paddings[i]).permute(1,0,2,3)
            temp2 = F.conv2d(neurons_2[i].permute(1,0,2,3),self.unpools[0](neurons_2[i+1], pools_idx_2[i]).permute(1,0,2,3),stride=1,padding=self.paddings[i]).permute(1,0,2,3)
            self.synapses[i].weight.grad = (temp1-temp2)/((betas[1]-betas[0])*x.size(0))
            temp1 = neurons_1[i+1].sum(dim=(0,2,3))
            temp2 = neurons_2[i+1].sum(dim=(0,2,3))
            self.synapses[i].bias.grad = (temp1-temp2)/((betas[1]-betas[0])*x.size(0))
        
        if criterion.__class__.__name__.find('MSE')==-1:
            pred = F.softmax(self.synapses[-1](neurons_1[-1].view(x.size(0),-1)),dim=1)-F.one_hot(y, num_classes=10)
            temp1 = -(torch.matmul(pred.T,neurons_1[-1].view(x.size(0),-1)) )

            pred = F.softmax(self.synapses[-1](neurons_2[-1].view(x.size(0),-1)),dim=1)-F.one_hot(y, num_classes=10)
            temp2 = -(torch.matmul(pred.T,neurons_2[-1].view(x.size(0),-1)) )
            self.synapses[-1].weight.grad = -(temp1+temp2)/(2*x.size(0))

            temp1 = F.softmax(self.synapses[-1](neurons_1[-1].view(x.size(0),-1)),dim=1)-F.one_hot(y, num_classes=10)
            temp2 =  F.softmax(self.synapses[-1](neurons_2[-1].view(x.size(0),-1)),dim=1)-F.one_hot(y, num_classes=10)
            
            self.synapses[-1].bias.grad = -(temp1+temp2).sum(dim=0)/(2*x.size(0))
 
# LCA Convolutional Neural Network
class LCA_CNN(torch.nn.Module):
    def __init__(self, device, in_size, channels, kernels, strides, 
                 fc, pools, unpools, paddings, activation=hard_sigmoid, softmax=False, scale_feedback=1.0, pretrain_dict=False, lca_params = None):
        
        super(LCA_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.fc = fc
        self.nc = fc[-1]        

        self.activation = activation
        self.pools = pools
        self.unpools = unpools
        self.poolidxs = []
        self.synapses = torch.nn.ModuleList()
        self.scale_feedback = scale_feedback
        self.softmax = softmax # whether to use softmax readout or not

        FEATURES = lca_params['lca_feats']  # number of dictionary features to learn
        KERNEL_SIZE = lca_params['lca_ksize']  # height and width of each feature
        LAMBDA = lca_params['lca_lambda']  # LCA threshold
        LCA_ITERS = lca_params['lca_iters'] # LCA iterations to run before adding feedback, reduced later
        LEARNING_RATE = lca_params['lca_eta']
        TAU = lca_params['lca_tau']
        STRIDE = lca_params['lca_stride']
        
        size = in_size # size of the input : 32 for cifar10
        self.lca = LCAConv2D(
                    out_neurons=FEATURES,
                    in_neurons=3,
                    result_dir='./dictionary_learning',
                    kernel_size=KERNEL_SIZE,
                    stride=STRIDE,
                    lambda_=LAMBDA,
                    tau=TAU,
                    track_metrics=False,
                    req_grad = False,
                    eta=LEARNING_RATE,
                    lca_iters = LCA_ITERS,
                    return_vars=['inputs', 'acts', 'recons', 'recon_errors','states'],
                )
        
        if pretrain_dict:
            pretrained_path = '/storage/jr3548@drexel.edu/eplcanet/results/CIFAR10/standard_dictlearning/'
            print('Loading pretrained dictionary from: ')
            ckpt = torch.load(os.path.join( pretrained_path, 'dictionary.pt'), map_location='cpu')
            self.lca.assign_weight_values(ckpt.weights)
        
        self.lca.to(device)

        size = size//self.lca.stride

        for idx in range(len(channels)-1): 
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], stride=strides[idx], padding=paddings[idx], bias=True))   
            size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool
        
        size = size * size * channels[-1]        
        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
        
        
    def init_poolidxs(self, x):
        
        mbs = x.size(0)
        device = x.device
        
        self.poolidxs = []
        append = self.poolidxs.append
        size = self.in_size
        
        size = size//self.lca.stride 
        
        for idx in range(len(self.channels)-1): 
            size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )   # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size),  device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]),  device=device))
        else:
            # we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), device=device))            
          
        return
    
    def init_neurons(self, x):
        mbs = x.size(0)
        device = x.device
        neurons = []
        append = neurons.append
        size = self.in_size
        
        size = size//self.lca.stride 
        
        for idx in range(len(self.channels)-1): 
            size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )   # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True,  device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True,  device=device))
                
        else:
            # we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            

        return neurons

    def Phi(self, x, y, acts, neurons, beta, criterion):
        conv_start = 1
        mbs = x.size(0)       
        conv_len = len(self.synapses) - 1
        tot_len = len(self.synapses) + 1


        layers = [acts] + neurons
        
        phi = 0#torch.zeros(x.shape[0], device=x.device, requires_grad=True)
        # Am I using the right pools?
        for idx in range(conv_start, conv_len): 
            phi = phi + torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()  # Shifted from setting LCA to layers[1]   
        for idx in range(conv_len, tot_len-1):
            phi = phi + torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
            
        # the prediction is made with softmax[last weights[penultimate layer]]
        if beta!=0.0:
            L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
            phi = phi - beta*L            
        
        return phi
    

    def forward(self, x, y=0, neurons=None, T=29, beta=0.0, 
                criterion=torch.nn.MSELoss(reduction='none'), scale_feedback=1.0, check_thm=False, frac = 1.0):
        global characteristic_param, characteristic_time, attack_param
        
        mbs = x.size(0)       
        conv_len = len(self.synapses) - 1
        tot_len = len(self.synapses)
        
        self.poolsidx = self.init_poolidxs(x)

        unpools = make_unpools('mmmm')
        
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = True

        # Do not include input activation layer in the system for now
        new_layers = self.init_neurons(x) 
        

        # Run lca to compute sparse code with pretrained dictionary 
        inputs,acts,recons,recon_errors,states = self.lca(x)

        layers = [acts] + neurons # LCA activations (as input) + RCNN states from previous step
        # for layer in new_layers:
        #     print('new', layer.shape)

        # for layer in layers:
        #     print(layer.shape)

        for t in range(T):
            cost = 0
            for idx in range(conv_len): # Feedforward
                new_layers[idx],self.poolidxs[idx] = self.pools[idx](self.synapses[idx](layers[idx])) # layers[0]: LCA activations 0-2
            
            # Feedback LCA Here?
            
            for idx in range(conv_len-1):  # Recurrent feedback (not including LCA activation input) 0-1
                # Not feeding back through layer 0 to LCA activation layer
                new_layers[idx] = new_layers[idx] + F.conv_transpose2d(unpools[idx](layers[idx+2],self.poolidxs[idx+1]),self.synapses[idx+1].weight,padding=self.paddings[idx+1])
            
            if(tot_len-conv_len!=1):
                # Feedback from higher FC to last conv
                new_layers[conv_len-1] = new_layers[conv_len-1] + torch.matmul(layers[conv_len+1],self.synapses[conv_len].weight).reshape(new_layers[conv_len-1].shape)

            if self.softmax: # FC layer not in neurons
                for idx in range(conv_len, tot_len-1): #-1 5, 5
                    new_layers[idx] = self.synapses[idx](layers[idx].view(mbs,-1)) #+ torch.matmul(layers[idx+1],self.synapses[idx+1].weight.T)
                for idx in range(conv_len, tot_len-2): # 5, 4 ?
                    new_layers[idx] = new_layers[idx] + torch.matmul(layers[idx+2],self.synapses[idx+1].weight)
                    
                if beta!=0.0:
                    y_hat = F.softmax(self.synapses[-1](layers[-1].view(x.size(0),-1)), dim = 1)  # Apply fc layer
                    cost = beta*torch.matmul((F.one_hot(y, num_classes=self.nc)-y_hat),self.synapses[-1].weight) # Compute cost from output
                    cost = cost.reshape(layers[-1].shape)
                    
            for idx in range(tot_len-1): 
                if idx==tot_len-2 and beta!=0:
                    layers[idx+1] = self.activation(new_layers[idx]+cost).detach()
                else:
                    layers[idx+1] = self.activation(new_layers[idx]).detach()
                
                layers[idx+1].requires_grad = True
        
        # RCNN neurons, pools, normalized lca inputs, lca activations
        return layers[1:],self.poolidxs, inputs,acts
    
    def compute_syn_grads_alternate(self, x, y, acts_1, acts_2, neurons_1, neurons_2,pools_idx_1, pools_idx_2, betas, criterion, check_thm=False):
        neurons_1 = [acts_1] + neurons_1
        neurons_2 = [acts_2] + neurons_2    
        
        # print('len(neurons_1)',len(neurons_1))
        # print('kernels',self.kernels)
        # for neuron in neurons_1:
        #     print(neuron.shape)
        
        for i in range(len(self.kernels)): # Does NOT update lca layer. Updates synapses only.
            temp1 = F.conv2d(neurons_1[i].permute(1,0,2,3),self.unpools[i](neurons_1[i+1], pools_idx_1[i].to(torch.int64)).permute(1,0,2,3),stride=1,padding=self.paddings[i]).permute(1,0,2,3)
            temp2 = F.conv2d(neurons_2[i].permute(1,0,2,3),self.unpools[i](neurons_2[i+1], pools_idx_2[i].to(torch.int64)).permute(1,0,2,3),stride=1,padding=self.paddings[i]).permute(1,0,2,3)
            
            self.synapses[i].weight.grad = (temp1-temp2)/((betas[1]-betas[0])*x.size(0)) # synapses[0]: first convolutional layer after lca
            
            # Compute bias gradient as sum of activity per sample
            temp1 = neurons_1[i+1].sum(dim=(0,2,3))
            temp2 = neurons_2[i+1].sum(dim=(0,2,3))
            
            self.synapses[i].bias.grad = (temp1-temp2)/((betas[1]-betas[0])*x.size(0)) 
        
        if criterion.__class__.__name__.find('MSE')==-1:
            pred = F.softmax(self.synapses[-1](neurons_1[-1].view(x.size(0),-1)),dim=1)-F.one_hot(y, num_classes=10)
            temp1 = -(torch.matmul(pred.T,neurons_1[-1].view(x.size(0),-1)) )

            pred = F.softmax(self.synapses[-1](neurons_2[-1].view(x.size(0),-1)),dim=1)-F.one_hot(y, num_classes=10)
            temp2 = -(torch.matmul(pred.T,neurons_2[-1].view(x.size(0),-1)) )
            self.synapses[-1].weight.grad = -(temp1+temp2)/(2*x.size(0))

            temp1 = F.softmax(self.synapses[-1](neurons_1[-1].view(x.size(0),-1)),dim=1)-F.one_hot(y, num_classes=10)
            temp2 =  F.softmax(self.synapses[-1](neurons_2[-1].view(x.size(0),-1)),dim=1)-F.one_hot(y, num_classes=10)
            
            self.synapses[-1].bias.grad = -(temp1+temp2).sum(dim=0)/(2*x.size(0))
            
    def compute_lca_update(self, inputs_2, acts_2, inputs_3, acts_3, betas):
        # LCA EP-like update (not used in pretrained code here)
        recon_2 = self.lca.compute_recon(acts_2,self.lca.weights)
        recon_error_2 = self.lca.compute_recon_error(inputs_2,recon_2)
        update_2 = self.lca.compute_weight_update(acts_2, recon_error_2)
        times_active_2 = compute_times_active_by_feature(acts_2) + 1
        
        recon_3 = self.lca.compute_recon(acts_3,self.lca.weights) 
        recon_error_3 = self.lca.compute_recon_error(inputs_3,recon_3)
        update_3 = self.lca.compute_weight_update(acts_3, recon_error_3) # update = acts . recon_error
        times_active_3 = compute_times_active_by_feature(acts_3) + 1 # num active coefficients per feature (for normalization)

        # Pretty much ends up as update_3 scaled (also might just need the nudged update based on some theory)
        update = 0.5*(update_2 / times_active_2 + update_3/ times_active_3) * self.lca.eta / (betas[1]-betas[0]) # * scale by -0.005
        #print('lca reconstruction update: ', update.sum())
        
        self.lca.update_weights_EP(update) # changed version to apply like lcapt package update_weights
        
        
    '''
    Add this function to _LCAConvBase class in lca.py from lca-pytorch library. I'll fix this later.
    
    def update_weights_EP(
        self, update: Tensor, normalize: bool = True, feedback: bool = False
    ) -> Tensor:
        """Updates the dictionary given the computed gradient"""
        with torch.no_grad():
            # Would clamping the LCA weight update help? (default inf) 
            update = torch.clamp(update, min=-self.d_update_clip, max=self.d_update_clip)
            
            # Add update to weights
            self.weights.copy_(self.weights + update)
            
            if feedback == False:
                if normalize:
                    self.normalize_weights()
                if self.lr_schedule is not None:
                    self.eta = self.lr_schedule(self.forward_pass)

            return update
    
    '''
        
def train(model, optimizer, train_loader, test_loader, T1, T2, betas, scale_feedback, device, epochs, criterion, alg='EP', 
          random_sign=False, save=False, check_thm=False, path='', checkpoint=None, thirdphase = False, scheduler=None, cep_debug=False,lca_check=False,lca_front=None,mean=0,std=0):
    print(model)
    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
    beta_1, beta_2 = betas
    norm_layer = Normalize(mean, std)

    df = pd.DataFrame(columns=['Train Accuracy', 'Validation Accuracy', 'Train Sparsity', 'Validation Sparsity', 'Train Reconstruction Error', 'Validation Reconstruction Error'])

    for epoch in range(epochs):
        run_correct = 0
        run_total = 0
        model.train()

        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            neurons = model.init_neurons(x)
            
            # First phase
            if isinstance(model, LCA_CNN):
                neurons, poolidxs, lca_inputs, lca_acts = model(x, y, neurons, T1, beta=beta_1, criterion=criterion,scale_feedback=scale_feedback)
            else:
                neurons, poolidxs = model(x, y, neurons, T1, beta=beta_1, criterion=criterion,scale_feedback=scale_feedback)
            
            neurons_1 = copy(neurons)

            # Predictions for running accuracy
            with torch.no_grad():
                if not model.softmax:
                    pred = torch.argmax(neurons[-1], dim=1).squeeze()
                else:
                    pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

                run_correct += (y == pred).sum().item()
                run_total += x.size(0)
                if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)) and save:
                    plot_neural_activity(neurons, path)
                    if isinstance(model,LCA_CNN):
                        plot_lca_weights(model, path + "lca_weights.png")

            # Second phase
            if random_sign and (beta_1==0.0):
                rnd_sgn = 2*np.random.randint(2) - 1
                betas = beta_1, rnd_sgn*beta_2
                beta_1, beta_2 = betas
                
            if (isinstance(model,LCA_CNN)):
                neurons,poolidxs_2, inputs_2, acts_2 = model(x, y, neurons, T2, beta = beta_2, criterion=criterion,scale_feedback=scale_feedback)
            else:
                neurons,poolidxs_2 = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                
            neurons_2 = copy(neurons)
            
            # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
            if thirdphase:
                #come back to the first equilibrium
                neurons = copy(neurons_1)
                if (isinstance(model,LCA_CNN)):
                    neurons,poolidxs_3, inputs_3, acts_3 = model(x, y, neurons, T2, beta = -beta_2, criterion=criterion,scale_feedback=scale_feedback)
                else:
                    neurons,poolidxs_3 = model(x, y, neurons, T2, beta = -beta_2, criterion=criterion, scale_feedback=scale_feedback)

                neurons_3 = copy(neurons)
                
                if not(isinstance(model, P_CNN)):
                    #model.compute_lca_update(inputs_2, acts_2, inputs_3, acts_3, (beta_2, - beta_2))
                    model.compute_syn_grads_alternate(x, y, acts_2, acts_3, neurons_2, neurons_3, poolidxs_2, poolidxs_3, (beta_2, - beta_2), criterion)
                else:
                    model.compute_syn_grads_alternate(x, y, neurons_2, neurons_3, poolidxs_2, poolidxs_3, (beta_2, - beta_2), criterion)
                    #model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
            else:
                model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)
                
            optimizer.step()      

            if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                run_acc = run_correct/run_total
                print('Epoch :', round(epoch+(idx/iter_per_epochs), 2),
                    '\tRun train acc :', round(run_acc,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                    timeSince(start, ((idx+1)+epoch*iter_per_epochs)/(epochs*iter_per_epochs)))
                plot_neural_activity(neurons, path)
                if isinstance(model, LCA_CNN):
                    plot_lca_weights(model, path + "lca_weights.png")
                    
        if scheduler is not None:
            if epoch < scheduler.T_max:
                scheduler.step()

        test_correct, test_acts, test_inputs = evaluate(model, test_loader, T1, device,mean=mean,std=std)
        test_acc = test_correct/(len(test_loader.dataset)) * 100
        
        train_acc = run_correct/run_total * 100
    
        if isinstance(model, LCA_CNN):
            train_recon = model.lca.compute_recon(acts_3,model.lca.weights)
            test_recon = model.lca.compute_recon(test_acts, model.lca.weights)
            train_recon_err = model.lca.compute_recon_error(inputs_3, train_recon).cpu().detach().numpy().mean()
            test_recon_err = model.lca.compute_recon_error(test_inputs, test_recon).cpu().detach().numpy().mean()
            train_sparsity = (acts_3 != 0).float().mean().item()
            test_sparsity = (test_acts != 0).float().mean().item()        
        else:
            train_recon_err = 0.0
            test_recon_err = 0.0
            train_sparsity = 0.0
            test_sparsity = 0.0

        torch.save(model, path + '/checkpoint.pt')


        epoch_df = pd.DataFrame({'Train Accuracy': [train_acc], 'Validation Accuracy': [test_acc], 
                            'Train Sparsity': [train_sparsity], 'Validation Sparsity': [test_sparsity],
                            'Train Reconstruction Error': [train_recon_err], 'Validation Reconstruction Error': [test_recon_err]})
        print(epoch_df)
        
        if epoch > 0:
            df = pd.concat([df, epoch_df], ignore_index=True, axis=0)
            plot_metrics(df['Train Accuracy'], df['Validation Accuracy'], 'Accuracy', path + 'accuracy.png')
            plot_metrics(df['Train Sparsity'], df['Validation Sparsity'], 'LCA Activation Sparsity', path + 'sparsity.png')
            plot_metrics(df['Train Reconstruction Error'], df['Validation Reconstruction Error'], 'Reconstruction Error', path + 'recon_error.png')  
        else:
            df = epoch_df
            
        df.to_csv(path + '/results.csv')
        torch.save(model, path + '/checkpoint.pt')
        
    df.to_csv(path + '/results.csv')
    torch.save(model, path + '/model.pt')


def evaluate(model, loader, T, device,mean=0,std=0):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    norm_layer = Normalize(mean, std)
    correct=0
    phase = 'Test'
    run_total = 0
    start_time = time.time()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        neurons = model.init_neurons(x)
        
        if isinstance(model, LCA_CNN):
            neurons, poolidxs, inputs_lca, acts_lca = model(x, y, neurons, T) # dynamics for T time steps
        else:
            neurons, poolidxs = model(x, y, neurons, T)
            acts_lca = torch.zeros_like(x)
            inputs_lca = torch.zeros_like(x)
            
        if not model.softmax:
            pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()
        correct += (y == pred).sum().item()
        run_total += 1
        
        # print(run_total,correct/(run_total*loader.batch_size),time.time()-start_time)
    acc = correct/len(loader.dataset) 
    print('\t'+phase+' accuracy :\t', acc)   
    
    return correct, acts_lca, inputs_lca
