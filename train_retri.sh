#!/bin/bash

source activate py37

CUDA_VISIBLE_DEVICES=0 python3 projects/FastRetri/train_net.py --config-file projects/FastRetri/config/cub.yml --num-gpus 1
CUDA_VISIBLE_DEVICES=0 python3 projects/FastRetri/train_net.py --config-file projects/FastRetri/config/cars.yml --num-gpus 1
CUDA_VISIBLE_DEVICES=0 python3 projects/FastRetri/train_net.py --config-file projects/FastRetri/config/sop.yml --num-gpus 1 
CUDA_VISIBLE_DEVICES=0 python3 projects/FastRetri/train_net.py --config-file projects/FastRetri/config/inshop.yml --num-gpus 1

