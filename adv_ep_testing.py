import os
import pandas as pd
import torch
from typing import List

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

from data_utils import *
from lca_utils_ep_fast import *

noise_levels = {1: 'C', 5: 'P'} # datasets with corrupted images at noise levels of severity 1 and 5

model_path = f'/storage/jr3548@drexel.edu/eplcanet/results/CIFAR10/EP/LCACNN/pretrained/2024-11-07/run_0'
data_path = f'/storage/jr3548@drexel.edu/eplcanet/data/corrupt_cifar10/CIFAR-10-{noise_levels[1]}'

model_params = read_hyperparameters(model_path + '/hyperparameters.txt')

model = torch.load(model_path + '/model.pt', map_location=torch.device(model_path['device']))

# evaluate model with black-box attacked imges
def corrupted_eval(model, loader, T, data_dir='./', results_dir='./results'):    
    model.eval()
    correct = 0
    mbs = 128
    corruption_accuracy = []
    norm_layer = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (3*0.2023, 3*0.1994, 3*0.2010))
    
    for p in ['gaussian_noise']:#, 'shot_noise', 'motion_blur', 'zoom_blur','spatter', 'brightness' ,'speckle_noise', 'gaussian_blur', 
              #'snow', 'contrast', 'defocus_blur','elastic_transform','fog','glass_blur','impulse_noise','jpeg_compression',
              #'pixelate','saturate','frost']:   
        avg_acc = []
        
        labels = torch.from_numpy(np.load(os.path.join(data_dir, 'labels.npy')))
        data = np.load(os.path.join(data_dir, p + '.npy'))
        data = torch.from_numpy(np.transpose(data,(0,3,1,2)))/255.
        dataset = torch.utils.data.TensorDataset(data, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=mbs, shuffle=False, num_workers=2, pin_memory=True)

        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            neurons = model(norm_layer(data))[-1] # shouldnt this specify T?
            # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons.view(data.size(0),-1)), dim = 1), dim = 1).squeeze()
            correct = (target == pred).cpu().numpy()
            avg_acc.append(correct)
        
        avg_acc = np.asarray(avg_acc)
        avg_acc = np.reshape(avg_acc,(5,(avg_acc.shape[1]*avg_acc.shape[0])//5))    
        avg_acc = avg_acc.mean(axis=1)
        corruption_accuracy.append(avg_acc)
        
    corruption_accuracy = np.asarray(corruption_accuracy)
    
    np.savetxt(results_dir + 'corrupt_acc.txt', corruption_accuracy,fmt='%.6f')
    
# Test model with black-box, corrupted images
corrupt_acc = corrupted_eval(model, x, y, attack_params, data_dir='./', results_dir='./results')
# TODO: implement white box attacks (ProjectedGradientDescent)... requires some changes to EP functions