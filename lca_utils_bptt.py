import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import os
from datetime import datetime
import time
import math
from data_utils import *

from itertools import repeat
from torch.nn.parameter import Parameter
from lcapt.lca import LCAConv2D

import collections
import matplotlib
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.estimators.classification import BlackBoxClassifierNeuralNetwork
from art.attacks.evasion.square_attack import SquareAttack
from art.attacks.evasion.auto_attack import AutoAttack
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
            pools.append( torch.nn.MaxUnpool2d(2, stride=2) )
        elif letters[p]=='a':
            pools.append( torch.nn.AvgPool2d(2, stride=2) )
        elif letters[p]=='i':
            pools.append( Identity2d(return_indices=False) )
    return pools


class LCA_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, unpools, paddings, activation=hard_sigmoid, softmax=False, lca=None, dict_loss="recon"):
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

        size = in_size # size of the input : 32 for cifar10

        self.synapses.append(lca)
        
        # LCA is changing the size and number of channels before EP
        size = int((size + 2 * paddings[0] - kernels[0]) / strides[0] + 1)  # size after lca layer
        print('size after lca: ', size)
        
        if self.pools[0].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[0].kernel_size)/pools[0].stride + 1 )
                
        print('size after lca, pool: ', size)
        for idx in range(1, len(channels)-1): 
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
                                                 stride=strides[idx], padding=paddings[idx], bias=True))
                
            size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool
            print('size: ', size)
        
        size = size * size * channels[-1]      
          
        fc_layers = [2048] + fc
        #print('size: ', size)

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
            if self.pools[idx].__class__.__name__.find('Pool')!=-1: # does not 
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            if idx == 0:
                size = 16
            append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True,  device=device))
        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True,  device=device))
        else:
            # we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            
    
        self.init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)

        return neurons

    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons
        phi = 0#torch.zeros(x.shape[0], device=x.device, requires_grad=True)

        for idx in range(len(self.pools)):
           self.pools[idx].return_indices = False
        
        #Phi computation changes depending on softmax == True or not
        if not self.softmax:
            for idx in range(conv_len):    
                phi = phi + torch.sum(self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
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
            self.synapses[0].lca_iters = 300
            lca_acts, recon_errors, states = self.synapses[0](layers[0])  # LCA called for 1 iteration, no external T=500
            self.synapses[0].lca_iters = 1
            
            phi = phi + torch.sum(self.pools[0](lca_acts) * layers[1], dim=(1,2,3)).squeeze()

            for idx in range(1, conv_len):
                phi = phi + torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     

            for idx in range(conv_len, tot_len-1):
                phi = phi + torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                phi = phi - beta*L            
        
        return phi

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), autograd=True, check_thm=False, return_lca=False):
 
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)   
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)  
        self.poolsidx = self.init_poolidxs(mbs,x.device)
        
        for idx in range(len(self.pools)):
           self.pools[idx].return_indices = True
        
        #poolidxs = [[] for i in range(len(self.pools))]
        layers = [x] + neurons
        new_layers = [] # tendency of neurons

        for neuron in neurons: # exclude input layer
            new_layers.append(torch.zeros_like(neuron, device=x.device))
        
        
        if autograd: # Forward function that I am using in train right now (9/29)
            for t in range(T):
                phi = self.Phi(x, y, neurons, beta, criterion)
                grads = torch.autograd.grad(phi, neurons, grad_outputs=self.init_grads, create_graph=False)

                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    neurons[idx].requires_grad = True

                if not_mse and not(self.softmax):
                    neurons[-1] = grads[-1]
                else:
                    neurons[-1] = self.activation( grads[-1] )

                neurons[-1].requires_grad = True
                
            if return_lca==True:
                # Will this change anything about the internal status of LCA and mess up learning?
                lca_acts, recon_error, states = self.synapses[0](layers[0])
                return neurons, lca_acts, recon_error
            else:
                return neurons
        
        else: # Could not get this (faster implementation) to work yet
             for t in range(T):
                if t == 0:
                #print('lca_acts.grad.sum(): ', layers[0].grad.sum())
                    lca_acts, recon_error, states = self.synapses[0](layers[0]) # states here = sparse code before threshold
                #print('lca_acts.sum(): ', lca_acts.sum())
                else:
                    lca_acts, recon_error, states = self.synapses[0](layers[0], initial_states=states) 
                
                #print((lca_acts != 0).float().mean())
                new_layers[0], self.poolidxs[0] = self.pools[0](lca_acts)
                
                #phi = self.Phi(x, y, neurons, beta, criterion)
                #init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                #grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)
                #print(grads[0].shape)
                
                for idx in range(1, conv_len):
                    #print('before conv: ', layers[idx].shape)
                    new_layers[idx],self.poolidxs[idx] = self.pools[idx](self.synapses[idx](layers[idx])) 

                # Don't need to update with convolutional transpose, right?
                
                #if self.softmax: # This might not be needed either
                #    for idx in range(conv_len, tot_len-1):
                #        new_layers[idx] = self.synapses[idx](layers[idx].view(mbs,-1)) #+ torch.matmul(layers[idx+1],self.synapses[idx+1].weight.T)
                #    for idx in range(conv_len, tot_len-2):
                #        new_layers[idx] = new_layers[idx] + torch.matmul(layers[idx+2],self.synapses[idx+1].weight)
                        
                for idx in range(tot_len-1):
                    layers[idx+1] = self.activation(new_layers[idx]).detach() # new_layers[0] becomes layers[1] for the next iteration
                    layers[idx+1].requires_grad = True
                
        if return_lca == True:
            return layers[1:], lca_acts, recon_error
        else:
            return layers[1:]
 
def train(model, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs, criterion, alg='EP', 
          random_sign=False, save=False, check_thm=False, path='', checkpoint=None, thirdphase = False, scheduler=None, cep_debug=False, dict_loss='class'):
    
    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
    beta_1, beta_2 = betas

    if checkpoint is None:
        train_recon_err, test_recon_err = [], []
        train_sparsity, test_sparsity = [], []
        train_acc = [10.0]
        test_acc = [10.0]
        best = 0.0
        epoch_sofar = 0
        angles = [90.0]
    else:
        train_recon_err, test_recon_err = (checkpoint["train_recon_err"], checkpoint["test_recon_err"],)
        train_sparsity, test_sparsity = (checkpoint["train_sparsity"], checkpoint["test_sparsity"])
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']    
        best = checkpoint['best']
        epoch_sofar = checkpoint['epoch']
        angles = checkpoint['angles'] if 'angles' in checkpoint.keys() else []

    for epoch in range(epochs):
        run_correct = 0
        run_total = 0
        model.train()

        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Normalize LCA weights?
            with torch.no_grad():
                model.synapses[0].normalize_weights()
                
            neurons = model.init_neurons(x.size(0), device)

            neurons = model(x, y, neurons, T1-T2, beta=0.0, criterion=criterion, autograd=True)           
            
            # detach data and neurons from the graph
            x = x.detach()
            x.requires_grad = True
            for k in range(len(neurons)):
                neurons[k] = neurons[k].detach()
                neurons[k].requires_grad = True

            neurons, lca_acts, recon_errors = model(x, y, neurons, T2, beta=0.0, criterion=criterion, autograd=True, return_lca=True) # T2 time step

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
                    plot_lca_weights(model, path + "lca_weights.png")
        
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
            
            log_gradients(model)
            optimizer.step()
        
            # Not returning LCA recon errors for now, will this compute it properly?
            
            if (dict_loss == "recon" or dict_loss == "combo"):  # Update LCA weights using activations and reconstructions from first phase  
                    #print(f'LCA weights before update_weights ({dict_loss}): ', model.synapses[0].weights[0][0][0])
                model.synapses[0].update_weights(lca_acts, recon_errors)

                    #print(f'LCA weights after update_weights ({dict_loss}): ', model.synapses[0].weights[0][0][0])
            
            #lca_acts = neurons[0].detach()
            #recon_errors = np.array([0.0]) # Filler until I figure out how to get LCA things properly
            
            lca_sparsity = (lca_acts != 0).float().mean().item()
            
            if (idx % (iter_per_epochs // 10) == 0) or (idx == iter_per_epochs - 1):
                run_acc = run_correct / run_total
                print("Epoch :",round(epoch_sofar + epoch + (idx / iter_per_epochs), 2),"\tRun train acc :", round(run_acc, 3),
                    "\t(" + str(run_correct) + "/" + str(run_total) + ")\t",timeSince(start, ((idx + 1) + epoch * iter_per_epochs) / (epochs * iter_per_epochs),))
                print(f"Avg recon error {recon_errors.mean()}\tActivation sparsity: {lca_sparsity}, ")
                if check_thm and alg!='BPTT':
                    BPTT, EP = check_gdu(model, x[0:5,:], y[0:5], T1, T2, betas, criterion, alg=alg)
                    RMSE(BPTT, EP)
    
        if scheduler is not None: # learning rate decay step
            if epoch+epoch_sofar < scheduler.T_max:
                scheduler.step()

        test_correct, test_acts, test_recon_errors = evaluate(model, test_loader, T1, device, autograd=True)# return_lca=True)
        test_acc_t = test_correct/(len(test_loader.dataset))
        
        mean_test_sparsity = (test_acts != 0).float().mean().item()
        
        if save:
            test_acc.append(100*test_acc_t)
            train_acc.append(100*run_acc)
            
            train_recon_err.append(recon_errors.mean().item())
            test_recon_err.append(test_recon_errors.mean().item())

            train_sparsity.append(lca_sparsity)
            test_sparsity.append(mean_test_sparsity)

            if test_correct > best:
                best = test_correct
                save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                            'train_acc': train_acc, 'test_acc': test_acc, 'train_recon_err': train_recon_err, 'test_recon_err': test_recon_err, 'train_sparsity': train_sparsity, 'test_sparsity': test_sparsity,
                            'best': best, 'epoch': epoch_sofar+epoch+1}
                save_dic['angles'] = angles
                save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                torch.save(save_dic,  path + '/checkpoint.tar')
                torch.save(model, path + '/model.pt')
            plot_lines(train_acc, test_acc, "Accuracy", "train", "test", "epoch", "accuracy", path + "/accuracy.png")
            plot_lines(train_recon_err, test_recon_err, "Recon Error", "train", "test", "epoch", "recon_error",path + "/recon_error.png")
            plot_lines(train_sparsity, test_sparsity, "Sparsity", "train", "test", "epoch", "sparsity", path + "/sparsity.png")     
    if save:
        save_dic = {
            "model_state_dict": model.state_dict(),
            "opt": optimizer.state_dict(),
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best": best,
            "epoch": epochs,
            "train_recon_err": train_recon_err,
            "test_recon_err": test_recon_err,
            "train_sparsity": train_sparsity,
            "test_sparsity": test_sparsity
        }
        save_dic["scheduler"] = (scheduler.state_dict() if scheduler is not None else None)
        torch.save(save_dic, path + "/final_checkpoint.tar")
        torch.save(model, path + "/final_model.pt")
 

def evaluate(model, loader, T, device, return_lca=False, autograd=False):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        if autograd:
            neurons, lca_acts, recon_errors = model(x, y, neurons, T, autograd=True, return_lca=True)  
        else:
            neurons, lca_acts, recon_errors = model(x, y, neurons, T, return_lca=True) # dynamics for T time steps

        if not model.softmax:
            pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 

    print(phase+' accuracy :\t', acc, ' recon errors: ', recon_errors.mean(), 'sparsity: ', (lca_acts != 0).float().mean().item())   
    return correct, lca_acts, recon_errors