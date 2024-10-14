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

        self.synapses = torch.nn.ModuleList()

        self.synapses.append(lca)
        

        # LCA is changing the size and number of channels before EP
        size = int(size/self.synapses[0].stride)
        #size = 32 # size after lca, CHANGE according to architecture (before we had hard coded)

                
        for idx in range(1, len(channels)-1): 
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
        
        size = int(size/self.synapses[0].stride)
        append(torch.zeros((mbs, self.channels[1], size, size),  device=device)) 
        
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
        for idx in range(len(self.channels)-1): # ACTUALLY, should size 0 be 32 and size 1 be 16? self.channels[0] is 3.
            if idx == 0:
                size = int(size/self.synapses[idx].stride)
                #size = 32 # size after lca, CHANGE according to architecture (hard coded)
            if idx > 0:
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
        # Energy computation
        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons
        phi = 0#torch.zeros(x.shape[0], device=x.device, requires_grad=True)

        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = False
        
        #Phi computation changes depending on softmax == True or not

        # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
        
        self.synapses[0].lca_iters = 300 # Run LCA to get activations
        lca_acts, recon_errors, states = self.synapses[0](layers[0])
        self.synapses[0].lca_iters = 1

        #print('sparsity: ', (lca_acts != 0).float().mean().item())
        
        #phi -= self.inhibitstrength*(F.conv_transpose2d(layers[idx+1], self.synapses[idx].weight, padding=self.synapses[idx].padding, stride=self.synapses[idx].stride, bias=None)).pow(2).sum(dim=(1,2,3))*0.5        
        #phi -= (F.conv_transpose2d(layers[1], self.synapses[0].weights, padding=(self.synapses[0].kernel_size // 2), stride=self.strides[0], bias=None)).pow(2).sum(dim=(1,2,3))*0.5        
        # LCA layer 
        #phi = phi + torch.sum(self.pools[0](lca_acts) * layers[1], dim=(1,2,3)).squeeze()

        phi = phi + torch.sum(self.pools[0](states) * layers[1], dim=(1,2,3)).squeeze()
        
        for idx in range(1, conv_len): # Convolutional layers
            phi = phi + torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
        
        for idx in range(conv_len, tot_len-1): # Fully connected layer
            phi = phi + torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
            
            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                phi = phi - beta*L            
        
        return phi
    
    #def lateral_phi(self, x, )

    # def forward(self, x, y=0, neurons=None, T=29, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False, return_lca=False):
    #     global characteristic_param, characteristic_time, attack_param
    #     if(attack_param == 1):
    #         neurons = self.init_neurons(x.size(0), x.device)
    #     not_mse = (criterion.__class__.__name__.find('MSE')==-1)
    #     mbs = x.size(0)       
    #     conv_len = len(self.kernels)
    #     tot_len = len(self.synapses)
    #     device = x.device     
    #     self.poolsidx = self.init_poolidxs(mbs,x.device)
    #     #unpools = make_unpools('immm')
    #     unpools = make_unpools('mmmmm')
        
    #     for idx in range(len(self.pools)):
    #         self.pools[idx].return_indices = True
        
    #     #poolidxs = [[] for i in range(len(self.pools))]
    #     layers = [x] + neurons
        
    #     new_layers = [] # tendency of neurons
    #     for neuron in neurons: # exclude input layer
    #         new_layers.append(torch.zeros_like(neuron, device=x.device))
    #     for t in range(T):
    #         #print('before lca: ', layers[0].shape)
    #         cost = 0 # ** Should I add recon_error to the cost?
    #         # LCA layer pass
    #         if t == 0:
    #             #print('lca_acts.grad.sum(): ', layers[0].grad.sum())
    #             lca_acts, recon_error, states = self.synapses[0](layers[0])
    #             # states is sparse code before threshold
    #             #print('lca_acts.sum(): ', lca_acts.sum())
    #         else:
    #             lca_acts, recon_error, states = self.synapses[0](layers[0], initial_states=states) 
    #             #print((lca_acts != 0).float().mean())
                
    #         #print('after lca: ', lca_acts.shape) 
            
    #         #new_layers[0], self.poolidxs[0] = self.pools[0](lca_acts) #new_layers[0] = lca_acts
    #         new_layers[0], self.poolidxs[0] = self.pools[0](states)
    #         #print('after pool: ', new_layers[0].shape)  

    #         # Convolutional layer passes
    #         for idx in range(1, conv_len):
    #             #print('before conv: ', layers[idx].shape)
    #             new_layers[idx],self.poolidxs[idx] = self.pools[idx](self.synapses[idx](layers[idx])) 
    #             #print('after conv, pool: ', new_layers[idx].shape)
 
    #         # Update LCA layer with EP feedback term from CNN layer (layer 1) to LCA layer (layer 0)
    #         #states = states + F.conv_transpose2d(unpools[0](layers[2], self.poolidxs[1]), self.synapses[1].weight, padding=self.paddings[1])

    #         for idx in range(1, conv_len-1): # skip first LCA layer
    #             new_layers[idx] = new_layers[idx] + F.conv_transpose2d(unpools[idx](layers[idx+2],self.poolidxs[idx+1]),self.synapses[idx+1].weight,padding=self.paddings[idx+1])
    #             #print(f'new_layers{idx}: ', new_layers[idx].mean())
    #             #print(f'layer{idx + 2} activation: ', layers[idx + 2].mean())
    #             #print(f'layer{idx + 2} conv transp:', F.conv_transpose2d(unpools[idx](layers[idx+2], self.poolidxs[idx + 1]), self.synapses[idx + 1].weight, padding=self.paddings[idx + 1]).mean())

    #         if(tot_len-conv_len!=1):
    #                 new_layers[conv_len-1] = new_layers[conv_len-1] + torch.matmul(layers[conv_len+1],self.synapses[conv_len].weight).reshape(new_layers[conv_len-1].shape)
    #         if self.softmax:
    #             for idx in range(conv_len, tot_len-1):
    #                 new_layers[idx] = self.synapses[idx](layers[idx].view(mbs,-1)) #+ torch.matmul(layers[idx+1],self.synapses[idx+1].weight.T)
    #             for idx in range(conv_len, tot_len-2):
    #                 new_layers[idx] = new_layers[idx] + torch.matmul(layers[idx+2],self.synapses[idx+1].weight)
    #             if beta!=0.0:
    #                 y_hat = F.softmax(self.synapses[-1](layers[-1].view(x.size(0),-1)), dim = 1)  
    #                 cost = beta*torch.matmul((F.one_hot(y, num_classes=self.nc)-y_hat),self.synapses[-1].weight)
    #                 cost = cost.reshape(layers[-1].shape)
    #         for idx in range(tot_len-1):
    #             if idx==tot_len-2 and beta!=0:
    #                 layers[idx+1] = self.activation(new_layers[idx]+cost).detach()
    #             else:
    #                 layers[idx+1] = self.activation(new_layers[idx]).detach() # new_layers[0] becomes layers[1] for the next iteration
                
    #             layers[idx+1].requires_grad = True
                
                
    #     if return_lca == True:
    #         return layers[1:], lca_acts, recon_error
    #     else:
    #         return layers[1:]

    def forward(self, x, y=0, neurons=None, T=29, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False, return_lca=False):
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device 
        
        for t in range(T):
                phi = self.Phi(x, y, neurons, beta, criterion)
                init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)

                
                
                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    neurons[idx].requires_grad = True
                
                neurons[-1] = self.activation( grads[-1] )

                neurons[-1].requires_grad = True

        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        # Make LCA connections symmetric, remove diagonal self connections
        #print(self.synapses[0].weights.shape) #64, 3, 7, 7
        
        #unlike the inter-layer connections, not each weight represents a unique pair of neurons.
                # the upper and lower triangle of this square matrix are the backwards and forwards connections
                # these need to be the same here.
        #with torch.no_grad():
        #    self.synapses[0].weights = torch.nn.Parameter(0.5 * (self.synapses[0].weights + self.synapses[0].weights.T) - torch.diag(torch.diagonal(self.synapses[0].weights) ))
    
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
            
            if alg=='EP':
                # First phase
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
            elif alg=='BPTT':
                neurons = model(x, y, neurons, T1-T2, beta=0.0, criterion=criterion)           
                # detach data and neurons from the graph
                x = x.detach()
                x.requires_grad = True
                for k in range(len(neurons)):
                    neurons[k] = neurons[k].detach()
                    neurons[k].requires_grad = True

                neurons, lca_acts, recon_errors = model(x, y, neurons, T2, beta=0.0, criterion=criterion, check_thm=True, return_lca=True) # T2 time step

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
            
            if alg=='EP':
                # Second phase
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas
                neurons = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                neurons_2 = copy(neurons)
                # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
                if thirdphase:
                    #come back to the first equilibrium
                    neurons = copy(neurons_1)
                    #neurons, lca_acts, recon_errors = model(x, y, neurons, T2, beta = - beta_2, criterion=criterion, return_lca=True)
                    neurons = model(x, y, neurons, T2, beta = - beta_2, criterion=criterion, return_lca=True)
                    neurons_3 = copy(neurons)
                    #log_gradients(model)
                    
                    model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                    lca_acts = torch.zeros_like(x)
                    recon_errors = torch.zeros_like(x)
                else:
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)
                #log_gradients(model)
                optimizer.step()      

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
                # log_gradients(model)
                # Backpropagation through time
                loss.backward()
                optimizer.step()
            
            if (dict_loss == "recon" or dict_loss == "combo"):  # Update LCA weights using activations and reconstructions from first phase  
                    #print(f'LCA weights before update_weights ({dict_loss}): ', model.synapses[0].weights[0][0][0])
                    model.synapses[0].update_weights(lca_acts, recon_errors)

                    #print(f'LCA weights after update_weights ({dict_loss}): ', model.synapses[0].weights[0][0][0])
            
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

        test_correct, test_acts, test_recon_errors = evaluate(model, test_loader, T1, device, return_lca=True)
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
 

def evaluate(model, loader, T, device, return_lca=False):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)

        neurons = model(x, y, neurons, T)#, return_lca=True) # dynamics for T time steps
        lca_acts, recon_errors = np.zeros_
        
        if not model.softmax:
            pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

        correct += (y == pred).sum().item()
    
    lca_acts = torch.zeros_like(x)
    recon_errors = torch.zeros_like(x)
                    
    acc = correct/len(loader.dataset) 
    if return_lca:
        print(phase+' accuracy :\t', acc, ' recon errors: ', recon_errors.mean(), 'sparsity: ', (lca_acts != 0).float().mean().item())   
        return correct, lca_acts, recon_errors
    else:
        print(phase+' accuracy :\t', acc) 
        return correct