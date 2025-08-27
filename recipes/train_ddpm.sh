#!/bin/bash

python train.py --config config/cifar10_ddpm --resume \
    --gpus '3,4'