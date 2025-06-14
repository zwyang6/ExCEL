import torch
import torch.nn as nn
from collections import OrderedDict


def reload_cpt(cpt_path):
    replace_keys = ['attn.qkv.weight','attn.qkv.bias', 'attn.proj.weight', 'attn.proj.bias']
    excel_keys = ['attn.in_proj_weight','attn.in_proj_bias','attn.out_proj.weight','attn.out_proj.bias']
    trained_state_dict = torch.load(cpt_path,map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        for new, old in zip(excel_keys, replace_keys):
            if old in k and 'resblocks' in k:
                k = k.replace(old, new)
            new_state_dict[k] = v


    return new_state_dict

if __name__ == "__main__":

    cpt_path = '/data/PROJECTS/MoRe_2024/codes/MoRe/00_exp/ablation_study/wo_dre/checkpoints/model_iter_20000.pth'
    gcr_path = '/home/jaye/Documents/PROJECTS/ModAL/00_gcn_based_codes/74.7_73.0_voc_sota_graphGCA_topk392_simplified/checkpoints/model_iter_2000.pth'
    save_path = cpt_path.replace('model_iter_20000.pth','model_iter_20000_gcr.pth')

    reload_cpt(cpt_path,gcr_path,save_path)
