#!/bin/bash


## Random noise for CIFAR10
python generate_random_pairs.py --scp '/home/btang5/work/2025/data/cifar10_raw/cifar10/scp/train/img.scp'\
    --output '/home/btang5/work/2025/pytorch-ddpm-ddp/data/cifar10_random_noise_from_true_images_train'\
    --img_size 32 \
    --ch 3