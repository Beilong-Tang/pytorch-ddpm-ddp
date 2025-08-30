# 
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DefaultConfig:
    # Unet
    ch:int = 128
    ch_mult:list = field(default_factory=lambda: [1, 2, 2,2])
    attn:list  =field(default_factory=lambda: [1])
    num_res_blocks:int =2
    dropout: float=0.1
    # Gaussian Diffusion
    beta_1: float =1.0e-4
    beta_T: float =0.02
    T:float =1000
    mean_type: str='epsilon'
    var_type: str='fixedlarge'
    # Training
    lr:float=2.0e-4
    grad_clip:int=1
    total_steps:int=800000
    img_size:int=32
    warmup:int=5000
    batch_size:int=128
    num_workers:int=4
    ema_decay:float=0.9999
    # Logging & Sampling
    logdir:str=None
    img_size:int=32
    sample_size:int=64
    save_step:int=5000
    sample_step:int=2500
    # Paired noise traing
    noise_scp:Optional[str]=None
    img_scp:Optional[str]=None
    ## Rectified Pretrained ckpt
    pretrained_ckpt:str=None
