
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import os
from torchvision.transforms import Compose, Normalize
from .decoder.TransDecoder import DecoderTransformer
import clip
from datasets.clip_text import new_class_names, BACKGROUND_CATEGORY,new_class_names_coco, BACKGROUND_CATEGORY_COCO
from .load_attr import attr_aggregate


class ExCEL_model(nn.Module):
    def __init__(self,  clip_model=None, embedding_dim=256, in_channels=512, dataset_name='pascal_voc', \
                        num_classes=21, num_atrr_clusters=112, json_file='./gpt4.0_cluster_a_photo_of4.json',\
                        img_size=320, mode='train', device='cuda'):

        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)
        self.encoder.visual.reload_self_attn(layers=6, feat_size=img_size//16, mode=mode)
        self.encoder.eval()
        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=12)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        text_prompts = new_class_names+BACKGROUND_CATEGORY if num_classes <= 21 else new_class_names_coco+BACKGROUND_CATEGORY_COCO
        self.integral_text_features = clip.encode_text_with_prompt_ensemble(self.encoder, text_prompts, device, prompt_templates=['a clean origami {}.'])
        self.text_attr, self.attr_flag = attr_aggregate(self.integral_text_features, dataset_name, num_classes-1, num_atrr_clusters, json_file)

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, img, ex_feats=None):

        if ex_feats is not None:
            image_features_, attn_weights_, all_feats_ = clip.generate_clip_fts(img, self.encoder, return_weights=True, ex_feats=ex_feats)
            attr_maps_raw_ = clip.clip_feature_surgery(image_features_, self.text_attr.permute(1,0))[:,1:,:self.num_classes-1]
            return attr_maps_raw_

        b, c, h, w = img.shape
        self.encoder.eval()
        image_features, attn_weights, all_feats = clip.generate_clip_fts(img, self.encoder, return_weights=True)
        attr_maps_raw = clip.clip_feature_surgery(image_features, self.text_attr.permute(1,0))[:,1:,:self.num_classes-1]
        # attr_maps_raw = clip.clip_feature_surgery(image_features, self.integral_text_features)[:,1:,:self.num_classes-1]

        all_img_tokens =  all_feats[:, :, 1:, ...]
        all_img_tokens = all_img_tokens.permute(0, 1, 3, 2)
        all_img_tokens = all_img_tokens.reshape(12, b, all_img_tokens.size(-2), h//16, w //16) #(11, b, c, h, w)

        fts = self.decoder_fts_fuse(all_img_tokens)
        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape
        
        seg, seg_attn_weight_list = self.decoder(fts)
        
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_fts_flatten = F.normalize(attn_fts_flatten, dim=1)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = (attn_pred - torch.mean(attn_pred) * 1.) * 3.0
        attn_pred = torch.sigmoid(attn_pred)

        return seg, attn_fts.clone().detach(), attr_maps_raw, attn_weights, attn_pred