#!/bin/bash

# Download pre-trained weights for evaluation
# Usage: bash download_weights.sh

set -e

echo "=========================================="
echo "Downloading pre-trained weights for ExCEL"
echo "=========================================="

# Create directories
mkdir -p ./00_sota/voc/checkpoints
mkdir -p ./00_sota/coco/checkpoints

# VOC checkpoint
echo "Downloading VOC checkpoint (model_iter_30000.pth)..."
VOC_URL="${VOC_URL:-https://drive.google.com/uc?id=VOC_FILE_ID}"
wget -q --show-progress -O ./00_sota/voc/checkpoints/model_iter_30000.pth "$VOC_URL"

# COCO checkpoint
echo "Downloading COCO checkpoint (model_iter_100000.pth)..."
COCO_URL="${COCO_URL:-https://drive.google.com/uc?id=COCO_FILE_ID}"
wget -q --show-progress -O ./00_sota/coco/checkpoints/model_iter_100000.pth "$COCO_URL"

echo "=========================================="
echo "Download complete!"
echo "=========================================="
