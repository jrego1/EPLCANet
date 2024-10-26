
This repository contains code to train an energy-based model (EBM) with a sparse coding frontend with equilibrium propagation and fine tune the sparse layer's dictionary for classification.

- main_fast.py: parses shell scripts, constructs model, optimizers, runs model from appropriate utils.py
- 
- pretrainedlca_utils_ep_fast.py: functions to build and train an energy-based recurrent convolutional neural network (RCNN) with a front sparse coding layer. The *pretrained* sparse dictionary is held fixed during EBM training with equilibrium propagation and preprocesses inputs with LCA dynamics.
- run/: shell scripts for running models and experiments
\\
\\
\\


*** *work in progress* ***

(TODO: Build and train an energy-based recurrent convolutional neural network (RCNN) with a front sparse coding layer using equilibrium propagation. *Tune* the sparse dictionary for classification by introducing feedback to activations during EBM training with equilibrium propagation.)

### LCA Sparse Coding PyTorch Implementation of the
This repository leverages functions from lca-pytorch.

https://github.com/lanl/lca-pytorch

### EqProp Implementation similar to "Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing its Gradient Estimator Bias"

This repository contains some code reworked from https://github.com/Laborieux-Axel/Equilibrium-Propagation, which produced the results of [the paper](https://arxiv.org/abs/2006.03824) "Scaling Equilibrium Prop to Deep ConvNets by Drastically Reducing its Gradient Estimator Bias". 

## Training

When setting the flags `--todo 'train' --save`, a results folder will be created at results/(EP or BPTT)/loss/yyyy-mm-dd/ with a plot of the train/test accuracy, reconstruction error, and dictionary sparsity updated at each epoch, a histogram of neural activations at each layer, and dictionary. The best performing model is saved at model.pt and the checkpoint for resuming training at checkpoint.tar. To resume training, simply rerun the same command line with the flag `--load-path 'results/.../hh-mm-ss'` and set the epoch argument to the remaining number of epochs. When the training is over, the final model and checkpoint are saved at final_model.pt and final_checkpoint.tar (they usually differ from the best model).

### Training an RCNN on CIFAR using equilibrium propagation with symmetric connections
. run/




## Summary table of command line arguments  

|Arguments|Description|Examples|
|-------|------|------|
|`model`|Choose MLP or CNN and Vector field.|`--model 'CNN'`, `--model 'LCACNN'`|
|`task`|Choose the task.|`--task 'MNIST'`, `--task 'CIFAR10'`|
|`data-aug`|Enable data augmentation for CIFAR10.|`--data-aug`|
|`lr-decay`|Enable learning rate decay.|`--lr-decay`|
|`scale`|Multiplication factor for weight initialization.|`--scale 0.2`|
|`channels`|Feature maps for CNN.|`--channels 128 256 512`|
|`pools`|Layers wise poolings. `m` is maxpool, `a` is avgpool and `i` is no pooling. All are kernel size 2 and stride 2.|`--pools 'mmm'` for 3 conv layers.|
|`kernels`|Kernel sizes for CNN.|`--kernels 3 3 3`|
|`strides`|Strides for CNN.|`--strides 1 1 1`|
|`paddings`|Padding for conv layers.|`--paddings 1 1 1`|
|`fc`|Linear classifier|`--fc 10` for one fc layer, `--fc 512 10`|
|`act`|Activation function for neurons|`--act 'tanh'`,`'mysig'`,`'hard_sigmoid'`|
|`todo`|Train or evaluate model|`--todo 'train'`,`--todo 'evaluate'`|
|`alg`|EqProp.|`--alg 'EP'`|
|`T1`,`T2`|Number of time steps for phase 1 and 2.|`--T1 30 --T2 10`|
|`betas`|Beta values beta1 and beta2 for EP phases 1 and 2.|`--betas 0.0 0.1`|
|`random-sign`|Choose a random sign for beta2.|`--random-sign`|
|`thirdphase`|Two phases 2 are done with beta2 and -beta2.|`--thirdphase`|
|`loss`|Loss functions.|`--loss 'mse'`,`--loss 'cel'`, `--loss 'cel' --softmax`|
|`optim`|Optimizer for training.|`--optim 'sgd'`, `--optim 'adam'`|
|`lrs`|Layer wise learning rates.|`--lrs 0.01 0.005`|
|`wds`|Layer wise weight decays. (`None` by default).|`--wds 1e-4 1e-4`|
|`mmt`|Global momentum. (if SGD).|`--mmt 0.9`|
|`lca_feats`|Number of LCA dictionary features.|`--lca_feats 64`|
|`lca_lambda`|LCA lambda.|`--lca_lambda 0.25`|
|`lca_tau`|LCA tau.|`--lca_tau 100`|
|`lca_eta`|LCA eta.|`--lca_eta 0.001`|
|`lca_stride`|LCA stride.|`--lca_stride 1`|
|`lca_iters`|LCA iterations.|`--lca_iters 1`|
|`dict_loss`|Dictionary update loss algorithm (reconstruction, classification, and both.|`--dict_loss 'recon'`|
|`scale_feedback`|Factor to scale feedback from convolutional layers to LCA activations for fine tuning.|`--scale_feedback 0.01`|
|`dict_training`|Static pretrained, fine-tune, or learn sparse coding dictionary during RCNN training. |` --dict_training pretrained_ep'`|
|`epochs`|Number of epochs.|`--epochs 200`|
|`mbs`|Minibatch size|`--mbs 128`|
|`device`|Index of the gpu.|`--device 0`|
|`save`|Create a folder to save the model, plot metrics, neural activations, and LCA weights.|`--save`|
|`load-path`|Resume the training of a saved simulations.|`--load-path 'results/2020-04-25/10-11-12'`|
|`seed`|Choose the seed.|`--seed 0`|
