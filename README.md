# [CVPR2025] Exploring CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation

We explore CLIP’s dense knowledge via a novel patch-text alignment paradigm for WSSS.

## News

* **If you find this work helpful, please give us a :star2: to receive the updation !**
* **` Feb. 26th, 2025`:** ExCEL is accepted by CVPR2025.
* **We are sorting the code...** 🔥🔥🔥

## Overview

<p align="middle">
<img src="/sources/main_fig.png" alt="ExCEL pipeline" width="1200px">
</p>

Weakly Supervised Semantic Segmentation (WSSS) with image-level labels aims to achieve pixel-level predictions using Class Activation Maps (CAMs). Recently, Contrastive Language-Image Pre-training (CLIP) has been introduced in WSSS. However, recent methods primarily focus on image-text alignment for CAM generation, while CLIP's potential in patch-text alignment remains unexplored. In this work, we propose ExCEL to explore CLIP's dense knowledge via a novel patch-text alignment paradigm for WSSS. Specifically, we propose Text Semantic Enrichment (TSE) and Visual Calibration (VC) modules to improve the dense alignment across both text and vision modalities. To make text embeddings semantically informative, our TSE module applies Large Language Models (LLMs) to build a dataset-wide knowledge base and enriches the text representations with an implicit attribute-hunting process. To mine fine-grained knowledge from visual features, our VC module first proposes Static Visual Calibration (SVC) to propagate fine-grained knowledge in a non-parametric manner. Then Learnable Visual Calibration (LVC) is further proposed to dynamically shift the frozen features towards distributions with diverse semantics. With these enhancements, ExCEL not only retains CLIP's training-free advantages but also significantly outperforms other state-of-the-art methods with much less training cost on PASCAL VOC and MS COCO.

## Data Preparation

### PASCAL VOC 2012

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
#### 2. Segmentation Labels

The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). The download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `VOCdevkit/VOC2012/`. 

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```

### MSCOCO 2014

#### 1. Download
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

#### 2. Segmentation Labels

To generate VOC style segmentation labels for COCO, you could use the scripts provided at this [repo](https://github.com/alicranck/coco2voc), or just download the generated masks from [Google Drive](https://drive.google.com/file/d/147kbmwiXUnd2dW9_j8L5L0qwFYHUcP9I/view?usp=share_link).

``` bash
COCO/
├── JPEGImages
│    ├── train2014
│    └── val2014
└── SegmentationClass
     ├── train2014
     └── val2014
```

## Requirement

Please refer to the requirements.txt. 

We incorporate a regularization loss for segmentation. Please refer to the instruction for this [python extension](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module).

## Train ExCEL
``` bash
### train voc
bash run_train.sh scripts/train_voc.py [gpu_device] [gpu_number] [master_port]  train_voc

### train coco
bash run_train.sh scripts/train_coco.py [gpu_devices] [gpu_numbers] [master_port] train_coco
```

## Evaluate ExCEL
``` bash
### eval voc seg and LAM
bash run_evaluate_voc.sh tools/infer_lam.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]

### eval coco seg
bash run_evaluate_seg_coco.sh tools/infer_seg_coco.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]
```

## Main Results

Semantic performance on VOC and COCO. Logs and weights are available now.
| Dataset | Backbone |  Val  | Test | Log |
|:-------:|:--------:|:-----:|:----:|:---:|
|   PASCAL VOC   |   ViT-B  | 78.4  | [78.5](http://host.robots.ox.ac.uk:8080/anonymous/1NNNE8.html) | [log](logs/voc_train.log) |
|   MS COCO  |   ViT-B  |  50.3 |   -  | [log](logs/coco_train.log) |

## Citation 
Please cite our work if you find it helpful to your reseach. :two_hearts:
```bash
@article{yang2024more,
  title={MoRe: Class Patch Attention Needs Regularization for Weakly Supervised Semantic Segmentation},
  author={Yang, Zhiwei and Neng, Yucong and Fu, Kexue and Wang, Shuo and Song, Zhijian},
  journal={arXiv preprint arXiv:2402.18467},
  year={2024}
}
```
If you have any questions, please feel free to contact the author by zwyang21@m.fudan.edu.cn.

## Acknowledgement
This repo is built upon [SeCo](https://github.com/zwyang6/SeCo.git) and [WeCLIP](https://github.com/zbf1991/WeCLIP). Many thanks to their brilliant works!!!
