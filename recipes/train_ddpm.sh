#!/bin/bash

python train.py --config config/cifar10_ddpm --resume \
    --gpus '0,1,2,3'