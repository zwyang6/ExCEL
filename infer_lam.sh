#!/bin/bash

file=./tools/infer_lam.py
device_gpu=0
nproc_per_node=1
master_port=29733

inferset=train

training_free=true

CUDA_VISIBLE_DEVICES=$device_gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file --infer_set=$inferset --training_free $training_free