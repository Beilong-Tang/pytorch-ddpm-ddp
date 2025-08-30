from .template import DefaultConfig

from dataclasses import dataclass

@dataclass
class Config(DefaultConfig):
    def __post_init__(self):
        self.pretrained_ckpt = "/home/btang5/work/2025/pytorch-ddpm/pretrained_ckpt/DDPM_CIFAR10_EPS/ckpt.pt"
        
        self.img_scp="/home/btang5/work/2025/pytorch-ddpm/sample_pairs/img_from_noise_pair/pretrained_ckpt/image.scp"
        self.noise_scp="/home/btang5/work/2025/pytorch-ddpm/sample_pairs/img_from_noise_pair/pretrained_ckpt/noise.scp"

        self.logdir="./logs/DDPM_CIFAR10_EPS_Rect_Diffusion_1"
