from .template import DefaultConfig

from dataclasses import dataclass

@dataclass
class Config(DefaultConfig):
    def __post_init__(self):
        self.img_scp="/home/btang5/work/2025/data/cifar10_raw/cifar10/scp/train/img.scp"
        self.noise_scp='/home/btang5/work/2025/pytorch-ddpm-ddp/data/cifar10_random_noise_from_true_images_train/noise.scp'

        self.logdir="./logs/DDPM_CIFAR10_EPS_random"
        
        self.num_workers=2