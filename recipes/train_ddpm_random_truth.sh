#!/bin/bash

# Train a DDPM random matching
# Note that the matching images are real from cifar 10, not from a pretrained DDPM

python train.py --config config/cifar10_ddpm_random_truth \
    --port 12344 \
    --gpus '5,6,7' \