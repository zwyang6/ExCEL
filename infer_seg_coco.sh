#!/bin/bash

inferset=val
crf=true

echo ******coco_with_crf
file=./tools/infer_seg_coco.py
cpt=./00_sota/coco/checkpoints/model_iter_100000.pth
python $file --model_path $cpt --infer_set $inferset  --crf_post $crf

echo ******coco_with_crf_evaluate_from_crf_predictions
file=./tools/infer_seg_coco_from_crf_pred.py
python $file
