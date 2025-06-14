import argparse
import os
import sys
import logging
sys.path.append("/ExCEL/")

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
from utils.reload import reload_cpt
from torch import multiprocessing
from utils.imutils import encode_cmap



parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="/ExCEL/w_outputs/voc/2024-09/voc_debug_cureattnpred_0.75caa2/checkpoints/model_iter_30000.pth", type=str, help="model_path")
parser.add_argument("--model", default="ExCEL_ViT-B/16", type=str, help="custom clip")
parser.add_argument("--dataset_name", default="pascal_voc", type=str, help="custom clip")
parser.add_argument("--attr_json", default="./attributes_text/descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json", type=str, help="custom clip")
parser.add_argument("--num_attri", default=112, type=int, help="number of attribution tokens")
parser.add_argument("--embedding_dim", default=256, type=int, help="number of attribution tokens")
parser.add_argument("--in_channels", default=768, type=int, help="number of attribution tokens")
parser.add_argument("--crf_post", default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--resize_long", default=512, type=int, help="resize the long side")
# parser.add_argument("--scales", default=[0.75, 1.0, 1.25, 1.5], help="multi_scales for seg")
parser.add_argument("--scales", default=[1.0], help="multi_scales for seg")

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
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--work_dir", default="results", type=str, help="work_dir")
parser.add_argument("--eval_set", default="val", type=str, help="infer_set")


def validate(model, dataset, test_scales=None):

    _preds, _gts, _msc_preds, cams = [], [], [], []
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    # with torch.no_grad(), torch.cuda.device(0):
    model.cuda(0)
    model.eval()

    num = 0

    _preds_hist = np.zeros((21, 21))
    _msc_preds_hist = np.zeros((21, 21))
    _cams_hist = np.zeros((21, 21))

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
        num+=1

        name, inputs, labels, cls_labels = data
        names = name+name

        inputs = inputs.cuda()
        labels = labels.cuda()

        #######
        # resize long side to 512
        _, _, h, w = inputs.shape
        ratio = args.resize_long / max(h,w)
        _h, _w = 320,320
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)
        #######

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        segs_cat = model(inputs_cat,)[0]
        
        segs = segs_cat[0].unsqueeze(0)

        _segs = (segs_cat[0,...] + segs_cat[1,...].flip(-1)) / 2
        _segs = F.interpolate(_segs.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False)[0]
        segs_list.append(_segs)

        # _, _, h, w = segs_cat.shape

        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
                segs_cat = model(inputs_cat,)[0]
                _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
                segs_list.append(_segs)

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)
        _preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        _gts += list(labels.cpu().numpy().astype(np.int16))


        if num % 100 == 0:
            _preds_hist, seg_score = evaluate.scores2(_gts, _preds, _preds_hist)
            _msc_preds_hist, msc_seg_score = evaluate.scores2(_gts, _msc_preds, _msc_preds_hist)
            _preds, _gts, _msc_preds, cams = [], [], [], []


        np.save(args.work_dir+ '/logit/' + name[0] + '.npy', {"segs":segs.detach().cpu().numpy(), "msc_segs":msc_segs.detach().cpu().numpy()})
            
    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist


def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.data_folder, 'JPEGImages',)
    labels_path = os.path.join(args.data_folder, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=64,  # 64
        bi_rgb_std=5,   # 5
        bi_w=4,         # 4
    )

    def _job(i):

        name = name_list[i]
        logit_name = os.path.join(args.work_dir, "logit", name + ".npy")

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.eval_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(os.path.join(args.work_dir, "prediction", name + ".png"), np.squeeze(pred).astype(np.uint8))
        imageio.imsave(os.path.join(args.work_dir, "prediction_cmap", name + ".png"), encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)
    hist = np.zeros((21, 21))
    hist, score = evaluate.scores2(gts, preds, hist, 21)

    print(score)
    
    return True

def main():

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
                        device='cuda')
    
    model.cuda()
    inputs  = torch.randn((4,3,320,320)).cuda()
    ssx = model(inputs)[0]
    trained_state_dict = torch.load(args.model_path,map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        if 'encoder.visual.positional_embedding' not in k:
            new_state_dict[k] = v
        else:
            pass
            # new_state_dict[k] = v


    # new_state_dict =reload_cpt(args.model_path)
    model.load_state_dict(state_dict=new_state_dict, strict=False)
    model.eval()

    gts, preds, msc_preds, cams, preds_hist, msc_preds_hist = validate(model=model, dataset=val_dataset, test_scales=[0.75, 1.0, 0.5, 1.25])
    torch.cuda.empty_cache()

    preds_hist, seg_score = evaluate.scores2(gts, preds, preds_hist)
    msc_preds_hist, msc_seg_score = evaluate.scores2(gts, msc_preds, msc_preds_hist)

    print("segs score:")
    print(seg_score)
    print("msc segs score:")
    print(msc_seg_score)

    crf_proc()

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)

    main()
