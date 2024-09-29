 python main_exp.py --model 'LCACNN' --task 'CIFAR10' --data-aug --channels 64 128 256 512 --kernels 5 3 3 3 --pools 'immm' --strides 2 1 1 1 --paddings 1 1 1 1 --fc 10 --optim 'sgd' --lrs 0.025 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 35 --act 'my_hard_sig' --todo 'train' --T1 300 --T2 50 --mbs 128 --alg 'EP' --betas 0.0 1.0 --thirdphase --loss 'cel' --softmax --save --lca_lambda 0.25 --tau 200 --eta 0.05 --lca_iters 1 --device 2 --dict_loss 'combo' --seed 2