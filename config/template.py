# 
from dataclasses import dataclass

@dataclass
class DefaultConfig:
    # Unet
    ch = 128
    ch_mult = [1,2,2,2]
    attn=[1]
    num_res_blocks=2
    dropout=0.1
    # Gaussian Diffusion
    beta_1=1.0e-4
    beta_T=0.02
    T=1000
    mean_type='epsilon'
    var_type='fixedlarge'
    # Training
    lr=2.0e-4
    grad_clip=1
    total_steps=800000
    img_size=32
    warmup=5000
    batch_size=128
    num_workers=4
    ema_decay=0.9999
    # Logging & Sampling
    logdir="./logs/DDPM_CIFAR10_EPS"
    img_size=32
    sample_size=64
    save_step=5000
    sample_step=2500
    # GPUs
    gpus=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]
    # Paired noise traing
    noise_scp=None
    img_scp=None
