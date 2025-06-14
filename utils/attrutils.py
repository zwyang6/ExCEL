import torch
import os

def load_text_attri(pt_path):
    type = os.path.basename(pt_path).replace('_cls_include4.pth','')
    attris = torch.load(pt_path,map_location='cpu')
    text_attri, attri_flag = attris[type]

    return text_attri.float().cuda(), attri_flag.cuda()

def attrmap2clsmap(attri_flag, attr_maps):

    attr_idx_mask = attri_flag.unsqueeze(0)

    cls_map = attr_maps @ attr_idx_mask.permute(0,2,1)

    return cls_map

def attr2cls_embedings(text_features, text_attri, num_classes):

    fg_text = text_features[:num_classes]
    bg_text = text_features[num_classes:]

    corr = (fg_text @ text_attri).softmax(dim=-1)
    text_attri_agg =corr @ text_attri.t() + text_features

    text_attri_agg = torch.cat([text_attri_agg,bg_text], dim=0)
    text_attri_agg = (text_attri_agg / text_attri_agg.norm(dim=1, keepdim=True)).permute(1,0)

    return text_attri_agg