This repository contains code to train an hybrid sparse coding, CNN (LCANet) using Equilibrium Propagation.

# PyTorch Implementation of the LCA Sparse Coding Algorithm
This repository uses functions from lca-pytorch.

https://github.com/lanl/lca-pytorch

# Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing its Gradient Estimator Bias

This repository contains some code reworked from https://github.com/Laborieux-Axel/Equilibrium-Propagation, which produced the results of [the paper](https://arxiv.org/abs/2006.03824) "Scaling Equilibrium Prop to Deep ConvNets by Drastically Reducing its Gradient Estimator Bias". 

## Training

When setting the flags `--todo 'train' --save`, a results folder will be created at results/(EP or BPTT)/loss/yyyy-mm-dd/ with a plot of the train/test accuracy, reconstruction error, and dictionary sparsity updated at each epoch, a histogram of neural activations at each layer, and dictionary. The best performing model is saved at model.pt and the checkpoint for resuming training at checkpoint.tar. To resume training, simply rerun the same command line with the flag `--load-path 'results/.../hh-mm-ss'` and set the epoch argument to the remaining number of epochs. When the training is over, the final model and checkpoint are saved at final_model.pt and final_checkpoint.tar (they usually differ from the best model).

### Training an LCANet CNN on MNIST using equilibrium propagation with symmetric connections
. check/train_mnist_eplcanet.sh

### Training an LCANet CNN on MNIST using standard back propagation
. check/train_mnist_bplcanet.sh


BPTT

```
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 30 --mbs 128 --alg 'BPTT' --loss 'cel' --softmax --save --device 0
```

## Evaluating

To evaluate a model, simply change the flag `--todo` to  `--todo 'evaluate'` and specify the path to the folder the same way as for resuming training. Train and Test accuracy will be appended to the hyperparameters.txt file.

```
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --todo 'evaluate' --T1 250 --mbs 200 --thirdphase --loss 'cel' --save --device 0 --load-path 'results/test'
```


## Comparing EP and BPTT

EP updates approximates ground truth gradients computed by BPTT. To check if the theorem is satisfied set the `--todo` flag to `--todo 'gducheck'`. With the flag `--save` enabled, plots comparing EP (dashed) and BPTT (solid) updates for each layers will be created in the results folder.

```
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --todo 'gducheck' --T1 250 --T2 15 --mbs 128 --thirdphase --betas 0.0 0.1 --loss 'cel' --save --device 0 --load-path 'results/test'
```

## More command lines

More command line are available at in the check folder of this repository, including training MLP on MNIST.
See the bottom of the page for a summary of all the arguments in the command lines.

## Summary table of the command lines arguments  

|Arguments|Description|Examples|
|-------|------|------|
|`model`|Choose MLP or CNN and Vector field.|`--model 'MLP'`, `--model 'VFMLP'`,`--model 'CNN'`,`--model 'VFCNN'`|
|`task`|Choose the task.|`--task 'MNIST'`, `--task 'CIFAR10'`|
|`data-aug`|Enable data augmentation for CIFAR10.|`--data-aug`|
|`lr-decay`|Enable learning rate decay.|`--lr-decay`|
|`scale`|Multiplication factor for weight initialisation.|`--scale 0.2`|
|`archi`|Layers dimension for MLP.|`--archi 784 512 10`|
|`channels`|Feature maps for CNN.|`--channels 128 256 512`|
|`pools`|Layers wise poolings. `m` is maxpool, `a` is avgpool and `i` is no pooling. All are kernel size 2 and stride 2.|`--pools 'mmm'` for 3 conv layers.|
|`kernels`|Kernel sizes for CNN.|`--kernels 3 3 3`|
|`strides`|Strides for CNN.|`--strides 1 1 1`|
|`paddings`|Padding for conv layers.|`--paddings 1 1 1`|
|`fc`|Linear classifier|`--fc 10` for one fc layer, `--fc 512 10`|
|`act`|Activation function for neurons|`--act 'tanh'`,`'mysig'`,`'hard_sigmoid'`|
|`todo`|Train or check the theorem|`--todo 'train'`,`--todo 'gducheck'`|
|`alg`|EqProp or BackProp Through Time.|`--alg 'EP'`, `--alg 'BPTT'`|
|`check-thm`|Check the theorem while training. (only if EP)|`--check-thm`|
|`T1`,`T2`|Number of time steps for phase 1 and 2.|`--T1 30 --T2 10`|
|`betas`|Beta values beta1 and beta2 for EP phases 1 and 2.|`--betas 0.0 0.1`|
|`random-sign`|Choose a random sign for beta2.|`--random-sign`|
|`thirdphase`|Two phases 2 are done with beta2 and -beta2.|`--thirdphase`|
|`loss`|Loss functions.|`--loss 'mse'`,`--loss 'cel'`, `--loss 'cel' --softmax`|
|`optim`|Optimizer for training.|`--optim 'sgd'`, `--optim 'adam'`|
|`lrs`|Layer wise learning rates.|`--lrs 0.01 0.005`|
|`wds`|Layer wise weight decays. (`None` by default).|`--wds 1e-4 1e-4`|
|`mmt`|Global momentum. (if SGD).|`--mmt 0.9`|
|`epochs`|Number of epochs.|`--epochs 200`|
|`mbs`|Minibatch size|`--mbs 128`|
|`device`|Index of the gpu.|`--device 0`|
|`save`|Create a folder where the accuracys are plotted upon training and the best model is saved.|`--save`|
|`load-path`|Resume the training of a saved simulations.|`--load-path 'results/2020-04-25/10-11-12'`|
|`seed`|Choose the seed.|`--seed 0`|
|`eps`|Adversarial Training Epsilon.|`--eps 0.05`|
|`image-print`|Print Attacked Images.|`--image-print 0`|
|`attack-step`|Timesteps used to generate perturbation.|`--attack-step 29`|
|`predict-step`|Timesteps used to generate perturbation.|`--predict-step 30`|
|`train-attack-step`|Adversarial Training Attack Timesteps.|`--train-attack-step 10`|
|`attack-norm`|Attack Norm.|`--attack-norm 2`|
|`n_feats`|Number of LCA dictionary features.|`--n_feats 784`|
|`lca_lambda`|LCA lambda.|`--lca_lambda 0.25`|
|`tau`|LCA tau.|`--tau 100`|
|`eta`|LCA eta.|`--eta 0.001`|
|`lca_stride`|LCA stride.|`--lca_stride 1`|
|`lca_iters`|LCA iterations.|`--lca_iters 1`|
|`dict_loss`|Dictionary update loss algorithm (reconstruction, classification, and both.|`--dict_loss 'recon'`|



