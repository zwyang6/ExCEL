#!/bin/bash

file=./scripts/train_voc.py
device_gpu=0
nproc_per_node=1
master_port=29733
exp_des=$1

CUDA_VISIBLE_DEVICES=$device_gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file --log_tag=$exp_des