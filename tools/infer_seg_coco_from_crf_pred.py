import argparse
import os
import sys
import logging
sys.path.append("./")

import imageio.v2 as imageio
from PIL import Image
import numpy as np
from datasets import coco
from tqdm import tqdm
from utils import evaluate
from utils.pyutils import format_tabs_multi_metircs, setup_logger

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="/data/PROJECTS/ExCEL_2024/ExCEL/00_sota/coco/checkpoints/model_iter_100000.pth", type=str, help="model_path")
parser.add_argument("--model", default="ExCEL_ViT-B/16", type=str, help="custom clip")
parser.add_argument("--dataset_name", default="ms_coco", type=str, help="custom clip")
parser.add_argument("--attr_json", default="./attributes_text/descriptors_ms_coco_gpt4.0_cluster_a_photo_of4.json", type=str, help="custom clip")
parser.add_argument("--num_attri", default=224, type=int, help="number of attribution tokens")
parser.add_argument("--embedding_dim", default=256, type=int, help="number of attribution tokens")
parser.add_argument("--in_channels", default=768, type=int, help="number of attribution tokens")
parser.add_argument("--crf_post", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--resize_size", default=320, type=int, help="resize the long side")
parser.add_argument("--scales", default=[0.7, 1.0, 1.2, 1.5], help="multi_scales for seg")
# parser.add_argument("--scales", default=[0.7, 1.0, 1.1, 1.2, 1.5], help="multi_scales for seg")

#! TO DO
## infer valset or testset: The test datafolder is different from valtrain folder
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--data_folder", default='/data/Datasets/MSCOCO2014/', type=str, help="dataset folder")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")


def validate_from_png(segs_path='/data/PROJECTS/ExCEL_2024/ExCEL/00_sota/coco/val/val_model_iter_100000_segs/seg_preds'):
    print("validating...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]


    images_path = os.path.join(args.data_folder, 'JPEGImages',)
    labels_path = os.path.join(args.data_folder, 'SegmentationClass')

    if "val" in args.infer_set:
        images_path = os.path.join(images_path, "val")
        labels_path = os.path.join(labels_path, "val")
    elif "train" in args.infer_set:
        images_path = os.path.join(images_path, "train")
        labels_path = os.path.join(labels_path, "train")

    gts, preds = [], []

    for name in tqdm(name_list, total=len(name_list), ncols=100, ascii=" >="):

        label_name = os.path.join(labels_path, name[13:] + ".png")
        # label = imageio.imread(label_name)
        label = np.asarray(Image.open(label_name))

        seg_name = os.path.join(segs_path, name + ".png")
        segs = imageio.imread(seg_name)
        
        gts.append(label)
        preds.append(segs)
        
    crf_score = evaluate.scores(gts, preds, num_classes=args.num_classes)
    logging.info('crf_seg_score:')
    metrics_tab_crf = format_tabs_multi_metircs([crf_score], ["confusion","precision","recall",'iou'], cat_list=coco.class_list)
    logging.info("\n"+ metrics_tab_crf)

    return True

if __name__ == "__main__":

    args = parser.parse_args()
    segs_path = './00_sota/coco/val/val_model_iter_100000_segs/seg_preds'
    log_name = segs_path.split('seg_preds')[0] + '/results_crf.log'
    setup_logger(filename=log_name)
    validate_from_png(segs_path=segs_path)
