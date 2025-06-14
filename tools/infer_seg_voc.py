import argparse
import os
import sys
import logging
sys.path.append("./")

from collections import OrderedDict
import imageio.v2 as imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc
from model.model_excel import ExCEL_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs_multi_metircs, setup_logger, convert_test_seg2RGB

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="/ExCEL/00_exp/00_ablation/main_ablation/baseline/checkpoints/model_iter_30000.pth", type=str, help="model_path")
parser.add_argument("--model", default="ExCEL_ViT-B/16", type=str, help="custom clip")
parser.add_argument("--dataset_name", default="pascal_voc", type=str, help="custom clip")
parser.add_argument("--attr_json", default="./attributes_text/descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json", type=str, help="custom clip")
parser.add_argument("--num_attri", default=112, type=int, help="number of attribution tokens")
parser.add_argument("--embedding_dim", default=256, type=int, help="number of attribution tokens")
parser.add_argument("--in_channels", default=768, type=int, help="number of attribution tokens")
parser.add_argument("--crf_post", default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--resize_size", default=320, type=int, help="resize the long side")
# parser.add_argument("--scales", default=[0.7, 1.0, 1.1, 1.2, 1.5], help="multi_scales for seg")
parser.add_argument("--scales", default=[0.7, 1.0, 1.2, 1.5], help="multi_scales for seg")
# parser.add_argument("--scales", default=[1.0], help="multi_scales for seg")

#! TO DO
## infer valset or testset: The test datafolder is different from valtrain folder
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--test_data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

def _validate(model=None, data_loader=None, args=None):

    model.eval()

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        gts, seg_pred = [], []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            _, _, h, w = inputs.shape
            seg_list = []

            _h, _w = args.resize_size, args.resize_size
            _inputs  = F.interpolate(inputs, size=[_h,_w], mode='bilinear', align_corners=False)
            inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
            segs = model(inputs_cat,)[0]
            segs = F.interpolate(segs, size=(h,w), mode='bilinear', align_corners=False)
            seg = segs[0].unsqueeze(0)
            seg_list.append(seg)

            for sc in args.scales:
                if sc != 1.0:
                    _h, _w = int(args.resize_size*sc), int(args.resize_size*sc)
                    _inputs  = F.interpolate(inputs, size=[_h,_w], mode='bilinear', align_corners=False)
                    inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
                    segs = model(inputs_cat,)[0]
                    segs = F.interpolate(segs, size=(h,w), mode='bilinear', align_corners=False)
                    seg = (segs[:1,...] + segs[1:,...].flip(-1))/2
                    seg_list.append(seg)

            segs = torch.mean(torch.stack(seg_list, dim=0), dim=0)
            # segs = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            seg_pred += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            if args.crf_post:
                np.save(args.logits_dir + "/" + name[0] + '.npy', {"msc_seg":segs.cpu().numpy()})
            elif args.infer_set == 'test':
                prob = F.softmax(segs, dim=1)[0].cpu().numpy()
                pred = np.argmax(prob, axis=0)
                convert_test_seg2RGB(np.squeeze(pred).astype(np.uint8),args.test_segs_dir + "/" + name[0] + ".png")

    seg_score = evaluate.scores(gts, seg_pred)
    logging.info('raw_seg_score:')
    metrics_tab = format_tabs_multi_metircs([seg_score], ["confusion","precision","recall",'iou'], cat_list=voc.class_list)
    logging.info("\n"+metrics_tab)
    return seg_score

def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.data_folder, 'JPEGImages',)
    labels_path = os.path.join(args.data_folder, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    # post_processor = DenseCRF(
    #     iter_max=10,    # 10
    #     pos_xy_std=3,   # 3
    #     pos_w=3,        # 3
    #     bi_xy_std=64,  # 64
    #     bi_rgb_std=5,   # 5
    #     bi_w=4,         # 4
    # )

    def _job(i):

        name = name_list[i]

        logit_name = args.logits_dir + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_seg']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.infer_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(args.segs_dir + "/" + name + ".png", np.squeeze(pred).astype(np.uint8))
        imageio.imsave(args.segs_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
    
        if args.infer_set == 'test':
            convert_test_seg2RGB(np.squeeze(pred).astype(np.uint8),args.test_segs_dir + "/" + name + ".png")
        
        return pred, label
    
    n_jobs = int(os.cpu_count() * 0.6)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds)
    logging.info('crf_seg_score:')
    metrics_tab_crf = format_tabs_multi_metircs([crf_score], ["confusion","precision","recall",'iou'], cat_list=voc.class_list)
    logging.info("\n"+ metrics_tab_crf)

    return crf_score

def validate(args=None):

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage=args.infer_set,
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)

    model = ExCEL_model(
                        clip_model=args.model, embedding_dim=args.embedding_dim, in_channels=args.in_channels, \
                        dataset_name=args.dataset_name, \
                        num_classes=args.num_classes, num_atrr_clusters=args.num_attri, json_file=args.attr_json,\
                        img_size=args.resize_size, mode=args.infer_set, device='cuda')

    model.cuda()
    trained_state_dict = torch.load(args.model_path,map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        if 'encoder.visual.positional_embedding' not in k:
            new_state_dict[k] = v

    model.load_state_dict(state_dict=new_state_dict, strict=False)
    model.eval()

    seg_score = _validate(model=model, data_loader=val_loader, args=args)
    torch.cuda.empty_cache()

    if args.crf_post:
        crf_score = crf_proc()
    
    return True

if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints/")[0] + f'/{args.infer_set}/'
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')
    args.logits_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/logits")
    args.segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/seg_preds")
    args.segs_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/seg_preds_rgb")

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    ### for test 
    if args.infer_set == 'test':
        crf = 'crf' if args.crf_post else 'no_crf'
        args.test_segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs_{crf}/results/VOC2012/Segmentation/comp6_test_cls/")
        os.makedirs(args.test_segs_dir, exist_ok=True)
        args.data_folder = args.test_data_folder
    
    setup_logger(filename=os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/results.log"))
    print(args)
    validate(args=args)
