import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import os

import time
import math
import pandas as pd
from data_utils import *
import matplotlib
from torchvision.transforms import Normalize


from lcapt.preproc import make_zero_mean, make_unit_var
from lcapt.metric import compute_times_active_by_feature

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
    
# LCA frontend Convolutional Neural Network

class LCA_CNN(torch.nn.Module):
    def __init__(self, device, in_size, channels, kernels, strides, fc, pools, unpools, paddings, activation=hard_sigmoid, softmax=False, scale_feedback=0.0,
                pretrain_dict=True, lca_params=None, finetune = False):
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
        
        self.finetune = finetune
        self.pretrained = pretrain_dict
        if lca_params is not None:

            FEATURES = lca_params['lca_feats']  # number of dictionary features to learn
            KERNEL_SIZE = lca_params['lca_ksize']  # height and width of each feature
            LAMBDA = lca_params['lca_lambda']  # LCA threshold
            LCA_ITERS = lca_params['lca_iters'] # LCA iterations to run before adding feedback, reduced later
            LEARNING_RATE = lca_params['lca_eta']
            TAU = lca_params['lca_tau']
            STRIDE = lca_params['lca_stride']
            
            print('Initializing LCA for preprocessing...')
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
            
            pretrained_path = '/storage/jr3548@drexel.edu/eplcanet/results/CIFAR10/standard_dictlearning/'
            print('Loading pretrained dictionary from: ')
            ckpt = torch.load(os.path.join( pretrained_path, 'dictionary.pt'), map_location='cpu')
            self.lca.assign_weight_values(ckpt.weights)
            
            self.lca.to(device)

            size = size//self.lca.stride
            
        # Add identity layer to include sparse activations in system
        self.synapses.append(torch.nn.Identity())

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
  
        if self.lca:
            size = size//self.lca.stride 
            # ** include LCA activations in system (?)
            append(torch.zeros((mbs, self.lca.out_neurons, size, size), requires_grad=True,  device=device))
        
        for idx in range(1, len(self.channels)-1): 
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
        if self.lca:
            size = size//self.lca.stride
            append(torch.zeros((mbs, self.lca.out_neurons, size, size), requires_grad=True,  device=device))
        
        for idx in range(1, len(self.channels)-1): 
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
    

    def forward(self, x, y=0, neurons=None, T=29, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), scale_feedback=1.0):
        global characteristic_param, characteristic_time, attack_param
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)       
        conv_len = len(self.synapses) - 1 #5
        tot_len = len(self.synapses) #6
   
        self.poolsidx = self.init_poolidxs(mbs,x.device)
        unpools = make_unpools('mmmmm')
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = True
        
        #poolidxs = [[] for i in range(len(self.pools))]
        inputs,acts,recons,recon_errors,states = self.lca(x)

        layers = [acts] + neurons
        
        new_layers = [] # tendency of neurons
        for neuron in neurons: # exclude input layer
            new_layers.append(torch.zeros_like(neuron, device=x.device))
            
        for t in range(T):
            cost = 0
            for idx in range(conv_len):
                new_layers[idx],self.poolidxs[idx] = self.pools[idx](self.synapses[idx](layers[idx])) 
            for idx in range(conv_len-1): 
                #else:
                new_layers[idx] = new_layers[idx] + F.conv_transpose2d(unpools[idx](layers[idx+2],self.poolidxs[idx+1]),self.synapses[idx+1].weight,padding=self.paddings[idx+1])
            if(tot_len-conv_len!=1):
                new_layers[conv_len-1] = new_layers[conv_len-1] + torch.matmul(layers[conv_len+1],self.synapses[conv_len].weight).reshape(new_layers[conv_len-1].shape)       
            
            if self.softmax:
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
        
        return layers[1:],self.poolidxs, inputs, acts
    
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

    def compute_syn_grads_alternate(self, x, y, acts_1, acts_2, neurons_1, neurons_2,pools_idx_1, pools_idx_2, betas, criterion, check_thm=False):
        #maybe include acts in neurons, return from forward
        neurons_1 = [acts_1] + neurons_1
        neurons_2 = [acts_2] + neurons_2

        for i in range(1, len(self.kernels)): # Don't update identity pass
            temp1 = F.conv2d(neurons_1[i].permute(1,0,2,3),self.unpools[i](neurons_1[i+1], pools_idx_1[i]).permute(1,0,2,3),stride=1,padding=self.paddings[i]).permute(1,0,2,3)
            temp2 = F.conv2d(neurons_2[i].permute(1,0,2,3),self.unpools[i](neurons_2[i+1], pools_idx_2[i]).permute(1,0,2,3),stride=1,padding=self.paddings[i]).permute(1,0,2,3)
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
            neurons, poolidxs, lca_inputs, lca_acts = model(x, y, neurons, T1, beta=beta_1, criterion=criterion,scale_feedback=scale_feedback)
            
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
                    
            # Second phase
            if random_sign and (beta_1==0.0):
                rnd_sgn = 2*np.random.randint(2) - 1
                betas = beta_1, rnd_sgn*beta_2
                beta_1, beta_2 = betas
                
            neurons,poolidxs_2, inputs_2, acts_2 = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                
            neurons_2 = copy(neurons)
            
            # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
            if thirdphase:
                #come back to the first equilibrium
                neurons = copy(neurons_1)
                neurons,poolidxs_3, inputs_3, acts_3 = model(x, y, neurons, T2, beta = -beta_2, criterion=criterion, scale_feedback=scale_feedback)

                neurons_3 = copy(neurons)

                model.compute_syn_grads_alternate(x, y, acts_2, acts_3,neurons_2, neurons_3, poolidxs_2, poolidxs_3, (beta_2, - beta_2), criterion)
                #model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                #else:    
                #    model.compute_syn_grads(x, y, neurons_1, neurons_2, (beta_2, - beta_2), criterion, neurons_3=neurons_3)
            else:
                model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)
                
            optimizer.step()      

            if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                run_acc = run_correct/run_total
                print('Epoch :', round(epoch+(idx/iter_per_epochs), 2),
                    '\tRun train acc :', round(run_acc,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                    timeSince(start, ((idx+1)+epoch*iter_per_epochs)/(epochs*iter_per_epochs)))
                plot_neural_activity(neurons, path)
                if model.finetune == True:
                    plot_lca_weights(model, path + 'lcaweights_epoch_' + str(epoch) + '.png')

        if model.finetune == True and model.pretrained == True: # Fine tune LCA weights with nudged, evolved state of activation layer once per epoch, not every batch like EP layers
            acts_star = model.lca.transfer(neurons_2[0])
            
            # Maybe I wan to run LCA dynamics once?
            # inhib = model.lca.lateral_competition(acts, connectivity)
            # states = states + (1 / model.lca.tau) * (input_drive - states - inhib)
            recon = model.lca.compute_recon(acts_star, model.lca.weights) 
            recon_error = model.lca.compute_recon_error(inputs_3,recon) # update = acts . recon_error
            model.lca.update_weights(acts_3, recon_error)

        elif model.finetune == True and model.pretrained == False: # Learn dictionary entirely during RCNN training
            acts_star = model.lca.transfer(neurons_2[0])
            
            # Maybe I wan to run LCA dynamics once?
            # inhib = model.lca.lateral_competition(acts, connectivity)
            # states = states + (1 / model.lca.tau) * (input_drive - states - inhib)
            recon = model.lca.compute_recon(acts_star, model.lca.weights) 
            recon_error = model.lca.compute_recon_error(inputs_3,recon) # update = acts . recon_error
            model.lca.update_weights(acts_3, recon_error)
            

        if scheduler is not None:
            if epoch < scheduler.T_max:
                scheduler.step()

        test_correct, test_inputs, test_acts = evaluate(model, test_loader, T1, device,mean=mean,std=std)
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
        
        neurons, poolidxs, inputs, acts = model(x, y, neurons, T)
            
        if not model.softmax:
            pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()
        correct += (y == pred).sum().item()
        run_total += 1
        
    acc = correct/len(loader.dataset) 
    print('\t'+phase+' accuracy :\t', acc)   
    
    return correct, inputs, acts

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