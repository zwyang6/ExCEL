#!/bin/bash

file=./tools/infer_seg_voc.py
inferset=train
crf=false

cpt=./00_sota/voc/checkpoints/model_iter_30000.pth
python $file --model_path $cpt --infer_set $inferset  --crf_post $crf