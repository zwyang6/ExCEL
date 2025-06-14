import argparse
import os
import sys
sys.path.append("./")
from collections import OrderedDict
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import logging
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc, coco
from model.model_excel import ExCEL_model
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from utils import evaluate, imutils
from utils.camutils import cure_attr_map, cure_attr_map_flip
from utils.affutils import refine_cams_with_aff, refine_cams_with_bkg_weclip

from utils.pyutils import format_tabs, setup_logger, format_tabs_multi_metircs,convert_test_seg2RGB
from utils.dcrf import DenseCRF
from utils.PAR import PAR

import joblib
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", default="./00_sota/voc/checkpoints/model_iter_30000.pth", type=str, help="model_path")
parser.add_argument("--model", default="ExCEL_ViT-B/16", type=str, help="custom clip")
parser.add_argument("--dataset_name", default="pascal_voc", type=str, help="custom clip")
parser.add_argument("--attr_json", default="./attributes_text/descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json", type=str, help="custom clip")
parser.add_argument("--num_attri", default=112, type=int, help="number of attribution tokens")
parser.add_argument("--embedding_dim", default=256, type=int, help="number of attribution tokens")
parser.add_argument("--in_channels", default=768, type=int, help="number of attribution tokens")
parser.add_argument("--resize_size", default=320, type=int, help="resize the long side")

parser.add_argument("--infer_set", default="train", type=str, help="infer_set")
parser.add_argument("--training_free", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--refine_with_aff", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="refine_cam_with_multiscale")
parser.add_argument("--crf_post", default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--save_cls_specific_cam", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="save the cam figs")
parser.add_argument("--save_cam", default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="save the cam figs")

#! TO DO
####refine_raw_CAM 2 masks with multiscales, if False output cam_seeds, else output refined_masks
parser.add_argument("--data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")

parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="work_dir")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size in training")

parser.add_argument("--nproc_per_node", default=8, type=int, help="nproc_per_node")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')


def build_validation(model=None, par=None, val_loader=None, device='cuda', args=None):

    gts, sms_attr_aff = [], [],
    color_map = plt.get_cmap("jet")

    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_labels = data
            img = imutils.denormalize_img(inputs)[0].permute(1,2,0).numpy()

            inputs  = F.interpolate(inputs, size=[args.resize_size, args.resize_size], mode='bilinear', align_corners=False)
            inputs = inputs.to(device, non_blocking=True)

            cls_labels = cls_labels.to(device, non_blocking=True)

            _, ex_feats, attr_maps_raw, attn_weights, attn_pred  = model(inputs)

            if args.training_free:
                # attr_maps_raw = cure_attr_map_flip(model, inputs, flip=True, ex_fts=False, raw_fts=None)
                attr_maps_raw = attr_maps_raw
            else:
                attr_maps_raw = cure_attr_map_flip(model, inputs)
                # attr_maps_raw = attr_maps_raw

            for i, attr_map in enumerate(attr_maps_raw):
                cls_label = cls_labels[i]
                attn_weight = attn_weights[:,i,...]
                seg_attn = attn_pred[i,...].unsqueeze(0)
                seg_attn = None if args.training_free else seg_attn
                refined_attr_maps, cls_lst = refine_cams_with_aff(attr_map, attn_weight, cls_label, size=inputs.shape[2:], seg_attn=seg_attn, caa_thre=0.79)
                attr_aff_labels, normed_attr_maps = refine_cams_with_bkg_weclip(refined_attr_maps, inputs[i], cls_lst, par, labels.shape[-2:])
                # imageio.imsave('/ExCEL/w_results/attr_aff' + "/" + name[0] + ".png", imutils.encode_cmap(np.squeeze(attr_aff_labels.cpu().numpy())).astype(np.uint8))

                if args.save_cam:
                    resized_attr_maps = normed_attr_maps[1:,...] # exclude bkg
                    cam_np = torch.max(resized_attr_maps, dim=0)[0].cpu().numpy()
                    cam_rgb = color_map(cam_np)[:,:,:3] * 255
                    alpha = 0.5
                    cam_rgb = alpha*cam_rgb + (1-alpha)*img
                    if not args.save_cls_specific_cam:
                        imageio.imsave(os.path.join(args.cam_dir, name[0] + ".jpg"), cam_rgb.astype(np.uint8))
                    else:
                        for cam,idx in zip(resized_attr_maps, cls_lst):
                            cam_np = cam.cpu().numpy()
                            cam_rgb = color_map(cam_np)[:,:,:3] * 255
                            alpha = 0.6
                            cam_rgb = alpha*cam_rgb + (1-alpha)*img
                            imageio.imsave(os.path.join(args.cs_cam_dir, name[0] + f"_{voc.class_list[idx+1]}.jpg"), cam_rgb.astype(np.uint8))
                
            sms_attr_aff += list(attr_aff_labels.cpu().numpy().astype(np.int16))        
            gts += list(labels.cpu().numpy().astype(np.int16))

            if args.crf_post:
                keys_gt = cls_lst
                valid_lam = normed_attr_maps
                np.save(args.logits_dir + "/" + name[0] + '.npy', {"valid_lam":valid_lam.cpu().numpy(),"keys_gt":keys_gt.cpu().numpy()})

    attr_aff_score = evaluate.scores(gts, sms_attr_aff, num_classes=args.num_classes)
    model.train()
    cat_list=voc.class_list if 'voc' in args.dataset_name else coco.class_list
    tab_results = format_tabs_multi_metircs([attr_aff_score], ["confusion","precision","recall",'iou'], cat_list=cat_list)
    logging.info(f'Training_free:{args.training_free}, LAM_score:')
    logging.info("\n"+tab_results)

    return tab_results

def validate(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage=args.infer_set,
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    model = ExCEL_model(
                        clip_model=args.model, embedding_dim=args.embedding_dim, in_channels=args.in_channels, \
                        dataset_name=args.dataset_name, \
                        num_classes=args.num_classes, num_atrr_clusters=args.num_attri, json_file=args.attr_json,\
                        img_size=args.resize_size, mode=args.infer_set, device='cuda')

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    if not args.training_free:
        new_state_dict = OrderedDict()
        for k, v in trained_state_dict.items():
            k = k.replace('module.', '')
            # new_state_dict[k] = v
            if 'encoder.visual.positional_embedding' not in k:
                new_state_dict[k] = v
        model.load_state_dict(state_dict=new_state_dict, strict=False)

    model.to(torch.device(args.local_rank))
    model.eval()

    model = DistributedDataParallel(model, device_ids=[args.local_rank],)
    n_gpus = dist.get_world_size()
    split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]
    val_loader = DataLoader(split_dataset[args.local_rank], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()

    results = build_validation(model=model, par=par, val_loader=val_loader, device='cuda', args=args)
    torch.cuda.empty_cache()

    if args.crf_post:
        crf_score = crf_proc()

    return True


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

    def _job(i):

        name = name_list[i]
        logit_name = args.logits_dir + "/" + name + ".npy"
        logit_ = np.load(logit_name, allow_pickle=True).item()
        lams = logit_['valid_lam']
        keys = logit_['keys_gt']


        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")

        if "test" in args.infer_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        prob = lams

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)
        keys = np.pad(keys+1, (1, 0), mode='constant')
        pred_crf = keys[pred].astype(np.uint8)
        imageio.imsave(args.segs_crf_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred_crf)).astype(np.uint8))

        return pred_crf,label

    
    n_jobs = int(os.cpu_count() * 0.6)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])
    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds)
    logging.info('crf_seg_score:')
    metrics_tab_crf = format_tabs_multi_metircs([crf_score], ["confusion","precision","recall",'iou'], cat_list=voc.class_list)
    logging.info("\n"+ metrics_tab_crf)

    return crf_score

if __name__ == "__main__":

    args = parser.parse_args()
    base_dir = args.model_path.split("checkpoints/")[0] + f'/{args.infer_set}/'
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')

    if args.training_free:
        tag = 'lam_training_free/aff_lam' if args.refine_with_aff else 'lam_training_free/seeds_lam'
    else:
        tag = 'lam_optimized/aff_lam' if args.refine_with_aff else 'lam_optimized/seeds_lam'

    if args.crf_post:
        args.logits_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_logits")
        args.segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/seg_preds")
        args.segs_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/seg_preds_rgb")
        args.segs_crf_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/segcrf_preds_rgb")
        os.makedirs(args.logits_dir, exist_ok=True)
        os.makedirs(args.segs_dir, exist_ok=True)
        os.makedirs(args.segs_rgb_dir, exist_ok=True)
        os.makedirs(args.segs_crf_rgb_dir, exist_ok=True)

    if args.save_cls_specific_cam:
        args.cs_cam_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_class_specific_img")
        os.makedirs(args.cs_cam_dir, exist_ok=True)

    args.cam_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_img")
    args.log_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_results.log")

    os.makedirs(args.cam_dir, exist_ok=True)
    setup_logger(filename=args.log_dir)
    logging.info('Pytorch version: %s' % torch.__version__)
    logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
    logging.info('\nargs: %s' % args)

    validate(args=args)