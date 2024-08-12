# Train LCANet with standard BP
    # dictionary loss: classification
python main.py --model 'EPLCACNN' --task 'MNIST' --alg 'BP' --channels 32 32 64 --kernels 5 5 --paddings 0 0 0 --strides 1 1 1 --optim 'adam' --lrs 1e-4 1e-5 8e-6 --pools 'imm' --epochs 20 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 3 --save --lca_lambda 0.25 --tau 200 --lca_iters 1 --softmax --thirdphase --dict_loss class --loss cel --seed 2
    # dictionary loss: reconstruction
python main.py --model 'EPLCACNN' --task 'MNIST' --alg 'BP' --channels 32 32 64 --kernels 5 5 --paddings 0 0 0 --strides 1 1 1 --optim 'adam' --lrs 1e-4 1e-5 8e-6 --pools 'imm' --epochs 20 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 3 --save --lca_lambda 0.25 --tau 200 --lca_iters 1 --softmax --thirdphase --dict_loss recon --loss cel --seed 2
    # dictionary loss: reconstruction and classification
python main.py --model 'EPLCACNN' --task 'MNIST' --alg 'BP' --channels 32 32 64 --kernels 5 5 --paddings 0 0 0 --strides 1 1 1 --optim 'adam' --lrs 1e-4 1e-5 8e-6 --pools 'imm' --epochs 20 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 3 --save --lca_lambda 0.25 --tau 200 --lca_iters 1 --softmax --thirdphase --dict_loss combo --loss cel --seed 2

# Train LCANet with EP
    # dictionary loss: classification
python main.py --model 'EPLCACNN' --task 'MNIST' --alg 'EP' --channels 32 32 64 --kernels 5 5 --paddings 0 0 0 --strides 1 1 1 --optim 'adam' --lrs 1e-4 1e-5 8e-6 --pools 'imm' --epochs 20 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 3 --save --lca_lambda 0.25 --tau 200 --lca_iters 1 --softmax --thirdphase --dict_loss class --loss cel --seed 2
    # dictionary loss: reconstruction
python main.py --model 'EPLCACNN' --task 'MNIST' --alg 'EP' --channels 32 32 64 --kernels 5 5 --paddings 0 0 0 --strides 1 1 1 --optim 'adam' --lrs 1e-4 1e-5 8e-6 --pools 'imm' --epochs 20 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 3 --save --lca_lambda 0.25 --tau 200 --lca_iters 1 --softmax --thirdphase --dict_loss recon --loss cel --seed 2
    # dictionary loss: reconstruction and classification
python main.py --model 'EPLCACNN' --task 'MNIST' --alg 'EP' --channels 32 32 64 --kernels 5 5 --paddings 0 0 0 --strides 1 1 1 --optim 'adam' --lrs 1e-4 1e-5 8e-6 --pools 'imm' --epochs 20 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 3 --save --lca_lambda 0.25 --tau 200 --lca_iters 1 --softmax --thirdphase --dict_loss combo --loss cel --seed 2
