import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

import os
from datetime import datetime
import time
import math
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






def make_pools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p]=='m':
            pools.append( torch.nn.MaxPool2d(2, stride=2,return_indices=True) )
        elif letters[p]=='a':
            pools.append( torch.nn.AvgPool2d(2, stride=2) )
        elif letters[p]=='i':
            pools.append( torch.nn.Identity() )
    return pools

def make_unpools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p]=='m':
            pools.append( torch.nn.MaxUnpool2d(2, stride=2) )
        elif letters[p]=='a':
            pools.append( torch.nn.AvgPool2d(2, stride=2) )
        elif letters[p]=='i':
            pools.append( torch.nn.Identity() )
    return pools
        


       
def my_init(scale):
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




        
   
    
    
# Convolutional Neural Network

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
    def init_neurons(self, mbs, device):
        
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

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons
        phi = 0#torch.zeros(x.shape[0], device=x.device, requires_grad=True)

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
    

    def forward(self, x, y=0, neurons=None, T=29, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        global characteristic_param, characteristic_time, attack_param
        if(attack_param == 1):
            neurons = self.init_neurons(x.size(0), x.device)
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)
        device = x.device     
        self.poolsidx = self.init_poolidxs(mbs,x.device)
        unpools = make_unpools('mmmm')
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = True
        
        #poolidxs = [[] for i in range(len(self.pools))]
        layers = [x] + neurons
        new_layers = [] # tendency of neurons
        for neuron in neurons: # exclude input layer
            new_layers.append(torch.zeros_like(neuron, device=x.device))
        for t in range(T):
            cost = 0
            for idx in range(conv_len):
                new_layers[idx],self.poolidxs[idx] = self.pools[idx](self.synapses[idx](layers[idx])) 
                #new_layers[idx] = self.pools[idx](self.synapses[idx](layers[idx]))
            for idx in range(conv_len-1):
                #h = w = layers[idx+2].shape[2]
                #upsampled = F.interpolate(layers[idx+2], [h*2, w*2], mode='nearest')
                #new_layers[idx] = new_layers[idx] +F.conv_transpose2d(upsampled,self.synapses[idx+1].weight,padding=self.paddings[idx+1])
                new_layers[idx] = new_layers[idx] + F.conv_transpose2d(unpools[idx](layers[idx+2],self.poolidxs[idx+1]),self.synapses[idx+1].weight,padding=self.paddings[idx+1])
            if(tot_len-conv_len!=1):
                    new_layers[conv_len-1] = new_layers[conv_len-1] + torch.matmul(layers[conv_len+1],self.synapses[conv_len].weight).reshape(new_layers[conv_len-1].shape)
            if self.softmax:
                for idx in range(conv_len, tot_len-1):
                    new_layers[idx] = self.synapses[idx](layers[idx].view(mbs,-1)) #+ torch.matmul(layers[idx+1],self.synapses[idx+1].weight.T)
                for idx in range(conv_len, tot_len-2):
                    new_layers[idx] = new_layers[idx] + torch.matmul(layers[idx+2],self.synapses[idx+1].weight)
                if beta!=0.0:
                    y_hat = F.softmax(self.synapses[-1](layers[-1].view(x.size(0),-1)), dim = 1)  
                    cost = beta*torch.matmul((F.one_hot(y, num_classes=self.nc)-y_hat),self.synapses[-1].weight)
                    cost = cost.reshape(layers[-1].shape)
            for idx in range(tot_len-1):
                if idx==tot_len-2 and beta!=0:
                    layers[idx+1] = self.activation(new_layers[idx]+cost).detach()
                else:
                    layers[idx+1] = self.activation(new_layers[idx]).detach()
                
                layers[idx+1].requires_grad = True
                
        # torch.save(self.poolidxs,f'pools_{int(beta*1000)}.pt')
        # torch.save(layers,f'layers_{int(beta*1000)}.pt')
        # torch.save(new_layers,f'new_layers_{int(beta*1000)}.pt')
        
        return layers[1:],self.poolidxs
    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = False
        self.zero_grad()            # p.grad is zero
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
        neurons_1 = [x]+neurons_1
        neurons_2 = [x] + neurons_2
        for i in range(len(self.kernels)):
            # print(neurons_1[i].size(),neurons_1[i+1].size())
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
    def __init__(self, in_size, channels, kernels, strides, 
                 fc, pools, unpools, paddings, activation=hard_sigmoid, softmax=False,
                 lca = None):
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
        
        self.softmax = softmax # whether to use softmax readout or not
        FEATURES = 64  # number of dictionary features to learn
        KERNEL_SIZE = 9  # height and width of each feature
        LAMBDA = 0.2  # LCA threshold
        LCA_ITERS = 1
        LEARNING_RATE = 1e-2
        TAU = 100
        STRIDE = 2
        size = in_size # size of the input : 32 for cifar10
        self.lca = LCAConv2D(
                    out_neurons=FEATURES,
                    in_neurons=3,
                    result_dir='./cifar_dictionary_learning',
                    kernel_size=KERNEL_SIZE,
                    stride=STRIDE,
                    lambda_=LAMBDA,
                    tau=TAU,
                    track_metrics=False,
                    req_grad = False,
                    eta=LEARNING_RATE,
                    lca_iters = 1,
                    return_vars=['inputs', 'acts', 'recons', 'recon_errors','states'],
                )
        
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
    def init_neurons(self, mbs, device):
        
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

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons
        phi = 0#torch.zeros(x.shape[0], device=x.device, requires_grad=True)

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
    

    def forward(self, x, y=0, neurons=None, T=29, beta=0.0, 
                criterion=torch.nn.MSELoss(reduction='none'), check_thm=False, frac = 1):
        global characteristic_param, characteristic_time, attack_param
        
        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)
        device = x.device     
        self.poolsidx = self.init_poolidxs(mbs,x.device)
        unpools = make_unpools('mmmm')
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = True
        
        inputs,acts,recons,recon_errors,states = self.lca(x)
        layers = [torch.zeros_like(acts)] + neurons
        new_layers = [] # tendency of neurons
        for neuron in neurons: # exclude input layer
            new_layers.append(torch.zeros_like(neuron, device=x.device))
        for t in range(T):
            cost = 0
            if t == 0:
                inputs,acts,recons,recon_errors,states = self.lca(x)
                layers[0] = states
               # layers[0] = F.relu(states - self.lca.lambda_) # Like thresholding, below threshold=0. (Here? Or after states are effected by convT) 
                
            else:
                inputs,acts,recons,recon_errors,states = self.lca(x,initial_states = states)
                
            for idx in range(conv_len):
                new_layers[idx],self.poolidxs[idx] = self.pools[idx](self.synapses[idx](layers[idx])) 
            
            # Feedback EP from layer+1 to membrane potentials (hopfield-like model, converge to steady state)
            states = frac*states + frac*F.conv_transpose2d(unpools[0](new_layers[0], self.poolidxs[0]),
                                                   self.synapses[0].weight,padding=1) #branch_point new_layers[0] or layers[1]
                                                    # synapses[0] between LCA-conv1 (new_layers[0])
            
            
            for idx in range(conv_len-1):
                new_layers[idx] = new_layers[idx] + F.conv_transpose2d(unpools[idx](layers[idx+2],self.poolidxs[idx+1]),self.synapses[idx+1].weight,padding=self.paddings[idx+1])
            if(tot_len-conv_len!=1):
                new_layers[conv_len-1] = new_layers[conv_len-1] + torch.matmul(layers[conv_len+1],self.synapses[conv_len].weight).reshape(new_layers[conv_len-1].shape)
            if self.softmax:
                for idx in range(conv_len, tot_len-1): # Apply FC layer
                    new_layers[idx] = self.synapses[idx](layers[idx].view(mbs,-1)) #+ torch.matmul(layers[idx+1],self.synapses[idx+1].weight.T)
                for idx in range(conv_len, tot_len-2):
                    new_layers[idx] = new_layers[idx] + torch.matmul(layers[idx+2],self.synapses[idx+1].weight)
                if beta!=0.0:
                    y_hat = F.softmax(self.synapses[-1](layers[-1].view(x.size(0),-1)), dim = 1)  
                    cost = beta*torch.matmul((F.one_hot(y, num_classes=self.nc)-y_hat),self.synapses[-1].weight)
                    cost = cost.reshape(layers[-1].shape)
            
            for idx in range(1, tot_len-1): # Don't apply activation to sparse layer (JR 10/11)
                if idx==tot_len-2 and beta!=0:
                    layers[idx+1] = self.activation(new_layers[idx]+cost).detach() # Add cost to last layer
                else:
                    layers[idx+1] = self.activation(new_layers[idx]).detach()
               
        if beta == 0:
            return layers[1:],self.poolidxs
        else:
            return layers, self.poolidxs, inputs, acts
        
    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = False
        self.zero_grad()            # p.grad is zero
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
        # neurons_1 = [x] + neurons_1
        # neurons_2 = [x] + neurons_2        
        for i in range(len(self.kernels)):
            # print(neurons_1[i].size(),neurons_1[i+1].size())
            # Compute weight gradient, difference in neuron activity of states, scaled by beta
            temp1 = F.conv2d(neurons_1[i].permute(1,0,2,3),self.unpools[0](neurons_1[i+1], pools_idx_1[i]).permute(1,0,2,3),stride=1,padding=self.paddings[i]).permute(1,0,2,3)
            temp2 = F.conv2d(neurons_2[i].permute(1,0,2,3),self.unpools[0](neurons_2[i+1], pools_idx_2[i]).permute(1,0,2,3),stride=1,padding=self.paddings[i]).permute(1,0,2,3)

            self.synapses[i].weight.grad = (temp1-temp2)/((betas[1]-betas[0])*x.size(0))
            
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
        recon_2 = self.lca.compute_recon(acts_2,self.lca.weights)
        recon_error_2 = self.lca.compute_recon_error(inputs_2,recon_2)
        update_2 = self.lca.compute_weight_update(acts_2, recon_error_2)
        times_active_2 = compute_times_active_by_feature(acts_2) + 1
        
        recon_3 = self.lca.compute_recon(acts_3,self.lca.weights)
        recon_error_3 = self.lca.compute_recon_error(inputs_3,recon_3)
        update_3 = self.lca.compute_weight_update(acts_3, recon_error_3) # upate = acts . recon_error
        times_active_3 = compute_times_active_by_feature(acts_3) + 1 # num active coefficients per feature (for normalization)
        
        update = (update_2 / times_active_2 - update_3/ times_active_3) * self.lca.eta / (betas[1]-betas[0])

        self.lca.update_weights_EP(update) # changed version to apply like lcapt package update_weights

        
def train(model, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs, criterion, alg='EP', 
          random_sign=False, save=False, check_thm=False, path='', checkpoint=None, thirdphase = False, scheduler=None, cep_debug=False,lca_check=False,lca_front=None,mean=0,std=0):
    
    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
    print("iterations per epoch",iter_per_epochs)
    print('LCA Check \t%d'%lca_check)
    beta_1, beta_2 = betas
    norm_layer = Normalize(mean, std)
    if checkpoint is None:
        train_acc = [10.0]
        test_acc = [10.0]
        best = 0.0
        epoch_sofar = 0
        angles = [90.0]
    else:
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']    
        best = checkpoint['best']
        epoch_sofar = checkpoint['epoch']
        angles = checkpoint['angles'] if 'angles' in checkpoint.keys() else []
    if (isinstance(model,LCA_CNN)):
        frac = 0.9
    else:
        frac = 1
    for epoch in range(epochs):
        run_correct = 0
        run_total = 0
        model.train()

        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            neurons = model.init_neurons(x.size(0), device)
            if alg=='EP' or alg=='CEP':
                # First phase
                neurons,poolidxs = model(x, y, neurons, T1, beta=beta_1, criterion=criterion,frac=frac)
                neurons_1 = copy(neurons)
            elif alg=='BPTT':
                neurons = model(x, y, neurons, T1-T2, beta=0.0, criterion=criterion)           
                # detach data and neurons from the graph
                x = x.detach()
                x.requires_grad = True
                for k in range(len(neurons)):
                    neurons[k] = neurons[k].detach()
                    neurons[k].requires_grad = True

                neurons = model(x, y, neurons, T2, beta=0.0, criterion=criterion, check_thm=True) # T2 time step

            # Predictions for running accuracy
            with torch.no_grad():
                if not model.softmax:
                    pred = torch.argmax(neurons[-1], dim=1).squeeze()
                else:
                    #WATCH OUT: prediction is different when softmax == True
                    pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

                run_correct += (y == pred).sum().item()
                run_total += x.size(0)
                if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)) and save:
                    plot_neural_activity(neurons, path)
            # print('EP calc,thirdphase',thirdphase)
            if alg=='EP':
                # Second phase
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas
                if (isinstance(model,LCA_CNN)):
                    neurons,poolidxs_2, inputs_2, acts_2 = model(x, y, neurons, T2, beta = beta_2, criterion=criterion,frac=frac)
                else:
                    neurons,poolidxs_2 = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                neurons_2 = copy(neurons)
                # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
                if thirdphase:
                    #come back to the first equilibrium
                    neurons = copy(neurons_1)
                    if (isinstance(model,LCA_CNN)):
                        neurons,poolidxs_3, inputs_3, acts_3 = model(x, y, neurons, T2, beta = -beta_2, criterion=criterion,frac=frac)
                    else:
                        neurons,poolidxs_3 = model(x, y, neurons, T2, beta = -beta_2, criterion=criterion)

                    neurons_3 = copy(neurons)
                    if not(isinstance(model, P_CNN)):
                        model.compute_lca_update(inputs_2, acts_2, inputs_3, acts_3, (beta_2, - beta_2))
                        model.compute_syn_grads_alternate(x, y, neurons_2, neurons_3, poolidxs_2, poolidxs_3, (beta_2, - beta_2), criterion)
                        
                        
                    else:
                        if model.same_update:
                            model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                        else:    
                            model.compute_syn_grads(x, y, neurons_1, neurons_2, (beta_2, - beta_2), criterion, neurons_3=neurons_3)
                        
                else:
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)
                    
                optimizer.step()      

            elif alg=='CEP':
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas

                # second phase
                if cep_debug:
                    prev_p = {}
                    for (n, p) in model.named_parameters():
                        prev_p[n] = p.clone().detach()
                    for i in range(len(model.synapses)):
                        prev_p['lrs'+str(i)] = optimizer.param_groups[i]['lr']
                        prev_p['wds'+str(i)] = optimizer.param_groups[i]['weight_decay']
                        optimizer.param_groups[i]['lr'] *= 6e-5
                        #optimizer.param_groups[i]['weight_decay'] = 0.0
                                        
                for k in range(T2):
                    neurons = model(x, y, neurons, 1, beta = beta_2, criterion=criterion)   # one step
                    neurons_2  = copy(neurons)
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)   # compute cep update between 2 consecutive steps 
                    for (n, p) in model.named_parameters():
                        p.grad.data.div_( (1 - optimizer.param_groups[int(n[9])]['lr']*optimizer.param_groups[int(n[9])]['weight_decay'])**(T2-1-k)  ) 
                    optimizer.step()                                                        # update weights 
                    neurons_1 = copy(neurons)
               
                if cep_debug:
                    debug(model, prev_p, optimizer)
 
                if thirdphase:    
                    neurons = model(x, y, neurons, T2, beta = 0.0, criterion=criterion)     # come back to s*
                    neurons_2 = copy(neurons)
                    for k in range(T2):
                        neurons = model(x, y, neurons, 1, beta = -beta_2, criterion=criterion)
                        neurons_3 = copy(neurons)
                        model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, -beta_2), criterion)
                        optimizer.step()
                        neurons_2 = copy(neurons)

            elif alg=='BPTT':
         
                # final loss
                if criterion.__class__.__name__.find('MSE')!=-1:
                    loss = 0.5*criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).mean().squeeze()
                else:
                    if not model.softmax:
                        loss = criterion(neurons[-1].float(), y).mean().squeeze()
                    else:
                        loss = criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).float(), y).mean().squeeze()
                # setting gradients field to zero before backward
                model.zero_grad()

                # Backpropagation through time
                loss.backward()

                optimizer.step()
                
            
            if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                run_acc = run_correct/run_total
                print('Epoch :', round(epoch_sofar+epoch+(idx/iter_per_epochs), 2),
                    '\tRun train acc :', round(run_acc,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                    timeSince(start, ((idx+1)+epoch*iter_per_epochs)/(epochs*iter_per_epochs)))
            # if isinstance(model, VF_CNN): 
            #     angle = model.angle()
            #     print('angles ',angle)
            # if check_thm and alg!='BPTT':
            #     BPTT, EP = check_gdu(model, x[0:5,:], y[0:5], T1, T2, betas, criterion, alg=alg)
            #     RMSE(BPTT, EP)
    
        if scheduler is not None: # learning rate decay step
            if epoch+epoch_sofar < scheduler.T_max:
                scheduler.step()

        test_correct = evaluate(model, test_loader, T1, device,lca_check,lca_front,mean=mean,std=std)
        test_acc_t = test_correct/(len(test_loader.dataset))
        if save:
            test_acc.append(100*test_acc_t)
            train_acc.append(100*run_acc)
            # if isinstance(model, VF_CNN):
            #     angle = model.angle()
            #     angles.append(angle)
            if test_correct > best:
                best = test_correct
                save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                            'train_acc': train_acc, 'test_acc': test_acc, 
                            'best': best, 'epoch': epoch_sofar+epoch+1}
                save_dic['angles'] = angles
                save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                torch.save(save_dic,  path + '/checkpoint.tar')
                torch.save(model, path + '/model.pt')
            plot_acc(train_acc, test_acc, path)        
    
    if save:
        save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                    'train_acc': train_acc, 'test_acc': test_acc, 
                    'best': best, 'epoch': epochs}
        save_dic['angles'] = angles
        save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
        torch.save(save_dic,  path + '/final_checkpoint.tar')
        torch.save(model, path + '/final_model.pt')

def evaluate(model, loader, T, device,lca_check=False,lca_front=None,mean=0,std=0):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    norm_layer = Normalize(mean, std)
    correct=0
    phase = 'est'
    run_total = 0
    start_time = time.time()
    print(len(loader.dataset),loader.batch_size,T,lca_check)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        neurons = model.init_neurons(x.size(0), device)
        neurons, poolidxs = model(x, y, neurons, T) # dynamics for T time steps

        if not model.softmax:
            pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

        correct += (y == pred).sum().item()
        run_total += 1
        
        # print(run_total,correct/(run_total*loader.batch_size),time.time()-start_time)
    acc = correct/len(loader.dataset) 
    print(phase+' accuracy :\t', acc)   
    return correct

    def parse_synset_mapping(path):
        """Parse the synset mapping file into a dictionary mapping <synset_id>:[<synonyms in English>]
        This assumes an input file formatted as:
            <synset_id> <category>, <synonym...>
        Example:
            n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
        """
        synset_map = []
        with open(path, 'r') as fp:
            lines = fp.readlines()
            counter = 0
            for line in lines:
                parts = line.split(' ')
                synset_map.append(parts[0])
                counter+=1
            return np.array(synset_map,dtype=str)
    # Evaluate the model on a dataloader with T steps for the dynamics
    map_clsloc_for_trainset = parse_synset_mapping('imagenet-32-batches-py/LOC_synset_mapping.txt')
    map_clsloc_for_corruptions = np.loadtxt('imagenet-32-batches-py/map_clsloc.txt',dtype=str)
    map_corruptions_to_trainset = {}
    for i in range(len(map_clsloc_for_corruptions)):
        idx = np.argwhere(map_clsloc_for_trainset==map_clsloc_for_corruptions[i][0])
        if(len(idx)!=1):
            print(idx,i,map_clsloc_for_corruptions[i][0])
        map_corruptions_to_trainset[idx.flatten()[0]]=i
    model.eval()
    correct=0
    phase = 'est'
    run_total = 0
    start_time = time.time()
    print(len(loader.dataset),loader.batch_size,T)
    # return 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        neurons = model(x, y, neurons, T) # dynamics for T time steps

        if not model.softmax:
            pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()
        correct += (pred.cpu().numpy()==np.vectorize(map_corruptions_to_trainset.get)(y.cpu().numpy())).sum()
        run_total += 1
        if(run_total%50==0):
            print(run_total,correct/(run_total*loader.batch_size),time.time()-start_time)
    acc = correct/len(loader.dataset) 
    print(phase+' accuracy :\t', acc)   
    return correct