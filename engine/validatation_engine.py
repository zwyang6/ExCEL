from utils.pyutils import AverageMeter, format_tabs
import torch
from tqdm import tqdm
from utils import evaluate
import numpy as np
import torch.nn.functional as F
from datasets import voc, coco
from utils import  evaluate
from utils.affutils import refine_cams_with_aff, refine_cams_with_bkg_weclip

def build_validation(model=None, par=None, val_loader=None, device='cuda', num_classes=21):

    gts, aff_pseudo, preds, sms_attr_aff = [], [], [], []    

    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_labels = data
            inputs  = F.interpolate(inputs, size=[320, 320], mode='bilinear', align_corners=False)

            inputs = inputs.to(device, non_blocking=True)
            cls_labels = cls_labels.to(device, non_blocking=True)

            segs, fts_diver, attr_maps_raw, attn_weights, attn_pred  = model(inputs)
            # attr_maps_raw = cure_attr_map(model,inputs, ex_feats=fts_diver)
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            for i, attr_map in enumerate(attr_maps_raw):
                cls_label = cls_labels[i]
                attn_weight = attn_weights[:,i,...]
                seg_attn = attn_pred[i,...].unsqueeze(0)
                refined_attr_maps, cls_lst = refine_cams_with_aff(attr_map, attn_weight, cls_label, size=inputs.shape[2:], seg_attn=seg_attn, caa_thre=0.75)
                attr_aff_labels,_ = refine_cams_with_bkg_weclip(refined_attr_maps, inputs[i], cls_lst, par, labels.shape[-2:])

            sms_attr_aff += list(attr_aff_labels.cpu().numpy().astype(np.int16))        
            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    attr_aff_score = evaluate.scores(gts, sms_attr_aff, num_classes=num_classes)
    seg_score = evaluate.scores(gts, preds, num_classes=num_classes)
    model.train()

    class_lst = voc.class_list if num_classes==21 else coco.class_list
    tab_results = format_tabs([attr_aff_score,seg_score], name_list=["Attr_aff_Pseudo","Seg_Preds"], cat_list=class_lst)

    return tab_results