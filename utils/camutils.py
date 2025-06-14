import torch
import torch.nn.functional as F
from .imutils import denormalize_img, encode_cmap
from .dcrf import crf_inference_label
import numpy as np
import imageio

def cure_attr_map_flip(model, inputs, ex_fts=True, flip=True, raw_fts=None):

    b, c, h, w = inputs.shape
    _h, _w = h//16, w//16

    with torch.no_grad():
        if flip:
            inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
            if ex_fts:
                ex_feats  = model(inputs_cat)[1]
                attr_maps_raw = model(inputs_cat, ex_feats=ex_feats)
            else:
                attr_maps_raw = model(inputs_cat)[2]
            _lam = attr_maps_raw.permute(0,2,1).reshape(b*2, -1, _h, _w)
            lam = torch.max(_lam[:b,...], _lam[b:,...].flip(-1))

            lam = lam + F.adaptive_max_pool2d(-lam, (1, 1))
            lam /= F.adaptive_max_pool2d(lam, (1, 1)) + 1e-5
            attr_maps_raw = lam.reshape(b,-1, _h*_w).permute(0,2,1)
        else:
            attr_maps_raw = model(inputs, ex_feats=raw_fts)

    return attr_maps_raw

def multi_scale_lam2(model, inputs, scales,cls_labels=None):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        segs, fts_diver, attr_maps_raw, attn_weights, attn_pred  =  model(inputs_cat,)
        _lam = attr_maps_raw
        _lam = F.interpolate(_lam, size=(h,w), mode='bilinear', align_corners=False)
        _lam = torch.max(_lam[:b,...], _lam[b:,...].flip(-1))

        lam_list = [_lam]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs, fts_diver, attr_maps_raw, attn_weights, attn_pred  =  model(inputs_cat,)
                _lam = attr_maps_raw

                _lam = F.interpolate(_lam, size=(h,w), mode='bilinear', align_corners=False)
                _lam = torch.max(_lam[:b,...], _lam[b:,...].flip(-1))

                lam_list.append(_lam)

        lam = torch.sum(torch.stack(lam_list, dim=0), dim=0)
        lam = lam + F.adaptive_max_pool2d(-lam, (1, 1))
        lam /= F.adaptive_max_pool2d(lam, (1, 1)) + 1e-5

    return lam
    
def multi_scale_lam(model, inputs, scales, crop_size):
    lam_list = []
    b, c, h, w = 1,0,crop_size,crop_size
    H,W = h //16, w//16
    n = 20
    with torch.no_grad():

        for s in scales:
            if s != 10.0:
                inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                # inputs = F.interpolate(inputs, size=(448, 448), mode='bilinear', align_corners=False)
                # inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
                inputs_cat = inputs

                _lam = model(inputs_cat, lam_only=True)
                # _lam = _lam.permute(0,2,1).reshape(b,n,448//16, 448//16)
                _lam = _lam.permute(0,2,1).reshape(b,n,int(s*H), int(s*W))
                _lam = F.interpolate(_lam, size=(h,w), mode='bilinear', align_corners=False)
                # _lam = torch.max(_lam[:b,...], _lam[b:,...].flip(-1))
                lam_list.append(_lam)

        lam = torch.sum(torch.stack(lam_list, dim=0), dim=0)
        lam = lam + F.adaptive_max_pool2d(-lam, (1, 1))
        lam /= F.adaptive_max_pool2d(lam, (1, 1)) + 1e-5
        # lam = lam.reshape(b,lam.shape[1],-1).permute(0,2,1)

    return lam

def cure_attr_map(model, inputs, ex_feats):
    with torch.no_grad():
        attr_maps_raw = model(inputs, ex_feats=ex_feats)

    return attr_maps_raw


def cam_to_label(cam, cls_label, img_box=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value<=cfg.cam.bkg_score] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=cfg.cam.high_thre] = cfg.dataset.ignore_index
        _pseudo_label[cam_value<=cfg.cam.low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return valid_cam, pseudo_label


def lam_to_label(cam, cls_label, img_box=None, bkg_thre=0.5, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None):
    b, c, h, w = cam.shape
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1

    if ignore_mid:
        _pseudo_label[cam_value <= high_thre] = ignore_index
        _pseudo_label[cam_value <= low_thre] = 0
    else:
        _pseudo_label[cam_value <= bkg_thre] = 0

    if img_box is not None:
        pseudo_label = torch.ones_like(_pseudo_label) * ignore_index
        for idx, coord in enumerate(img_box):
            pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]
    else:
        pseudo_label = _pseudo_label

    return valid_cam, pseudo_label

def ignore_img_box(label, img_box, ignore_index):

    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label

def cam_to_fg_bg_label(imgs, cams, cls_label, bg_thre=0.3, fg_thre=0.6):

    scale = 2
    imgs = F.interpolate(imgs, size=(imgs.shape[2]//scale, imgs.shape[3]//scale), mode="bilinear", align_corners=False)
    cams = F.interpolate(cams, size=imgs.shape[2:], mode="bilinear", align_corners=False)

    b, c, h, w = cams.shape
    _imgs = denormalize_img(imgs=imgs)

    cam_label = torch.ones(size=(b, h, w),).to(cams.device)
    bg_label = torch.ones(size=(b, 1),).to(cams.device)
    _cls_label = torch.cat((bg_label, cls_label), dim=1)

    lt_pad = torch.ones(size=(1, h, w),).to(cams.device) * bg_thre
    ht_pad = torch.ones(size=(1, h, w),).to(cams.device) * fg_thre

    for i in range(b):
        keys = torch.nonzero(_cls_label[i,...])[:,0]
        #print(keys)
        n_keys = _cls_label[i,...].cpu().numpy().sum().astype(np.uint8)
        valid_cams = cams[i, keys[1:]-1, ...]
        
        lt_cam = torch.cat((lt_pad, valid_cams), dim=0)
        ht_cam = torch.cat((ht_pad, valid_cams), dim=0)

        _, cam_label_lt = lt_cam.max(dim=0)
        _, cam_label_ht = ht_cam.max(dim=0)
        #print(_imgs[i,...].shape)
        _images = _imgs[i,...].permute(1,2,0).cpu().numpy().astype(np.uint8)
        _cam_label_lt = cam_label_lt.cpu().numpy()
        _cam_label_ht = cam_label_ht.cpu().numpy()
        _cam_label_lt_crf = crf_inference_label(_images, _cam_label_lt, n_labels=n_keys)
        _cam_label_lt_crf_ = keys[_cam_label_lt_crf]
        _cam_label_ht_crf = crf_inference_label(_images, _cam_label_ht, n_labels=n_keys)
        _cam_label_ht_crf_ = keys[_cam_label_ht_crf]
        #_cam_label_lt_crf = torch.from_numpy(_cam_label_lt_crf).to(cam_label.device)
        #_cam_label_ht_crf = torch.from_numpy(_cam_label_ht_crf).to(cam_label.device)
        
        cam_label[i,...] = _cam_label_ht_crf_
        cam_label[i, _cam_label_ht_crf_==0] = 255
        cam_label[i, (_cam_label_ht_crf_ + _cam_label_lt_crf_)==0] = 0
        #imageio.imsave("out.png", encode_cmap(cam_label[i,...].cpu().numpy()))
        #cam_label_lt

    return cam_label

def multi_scale_cam(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam, _ = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        
        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam, _ = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam

def multi_scale_cam_with_aff_mat(model, inputs, scales):
    cam_list, aff_mat = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam, _aff_mat = model(inputs_cat, cam_only=True)
        aff_mat.append(_aff_mat)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        
        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam, _aff_mat = model(inputs_cat, cam_only=True)
                aff_mat.append(_aff_mat)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

    max_aff_mat = aff_mat[np.argmax(scales)]
    return cam, max_aff_mat

def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, img_box=None, down_scale=2):

    b,_,h,w = images.shape
    _images = F.interpolate(images, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b,1,h,w))*cfg.cam.high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b,1,h,w))*cfg.cam.low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * cfg.dataset.ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()
    
    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)

    for idx, coord in enumerate(img_box):

        valid_key = torch.nonzero(cls_labels[idx,...])[:,0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))
        
        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1], coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1], coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = cfg.dataset.ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label

def refine_cams_with_cls_label(ref_mod=None, images=None, labels=None, cams=None, img_box=None):
    
    refined_cams = torch.zeros_like(cams)
    b = images.shape[0]

    #bg_label = torch.ones(size=(b, 1),).to(labels.device)
    cls_label = labels

    for idx, coord in enumerate(img_box):

        _images = images[[idx], :, coord[0]:coord[1], coord[2]:coord[3]]

        _, _, h, w = _images.shape
        _images_ = F.interpolate(_images, size=[h//2, w//2], mode="bilinear", align_corners=False)

        valid_key = torch.nonzero(cls_label[idx,...])[:,0]
        valid_cams = cams[[idx], :, coord[0]:coord[1], coord[2]:coord[3]][:, valid_key,...]

        _refined_cams = ref_mod(_images_, valid_cams)
        _refined_cams = F.interpolate(_refined_cams, size=_images.shape[2:], mode="bilinear", align_corners=False)

        refined_cams[idx, valid_key, coord[0]:coord[1], coord[2]:coord[3]] = _refined_cams[0,...]

    return refined_cams


def cams_to_affinity_label(cam_label, mask=None, ignore_index=255):
    
    b,h,w = cam_label.shape
    
    cam_label_resized = F.interpolate(cam_label.unsqueeze(1).type(torch.float32), size=[h//16, w//16], mode="nearest")

    h,w = h//16, w//16

    # cam_label_resized = F.interpolate(cam_label.unsqueeze(1).type(torch.float32), size=[h//8, w//8], mode="nearest")

    _cam_label = cam_label_resized.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0,2,1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    #aff_label[(_cam_label_rep+_cam_label_rep_t) == 0] = ignore_index
    for i in range(b):

        if mask is not None:
            aff_label[i, mask==0] = ignore_index

        aff_label[i, :, _cam_label_rep[i, 0, :]==ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :]==ignore_index, :] = ignore_index
    aff_label[:, range(h*w), range(h*w)] = ignore_index

    return aff_label

def propagte_aff_cam(cams, aff=None, mask=None):
    b, c, h, w = cams.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask==0] = 0

    #cams = F.interpolate(cams, size=[h//16, w//16], mode="bilinear", align_corners=False).detach()
    cams_rw = cams.clone()

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-4)

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _cams = cams[i].reshape(c, -1)
        _aff = aff[i]
        _cams_rw = torch.matmul(_cams, _aff)
        cams_rw[i] = _cams_rw.reshape(cams_rw[i].shape)

    #cams_rw = F.interpolate(cams_rw, size=[h, w], mode="bilinear", align_corners=False)

    return cams_rw

def propagte_aff_cam_with_bkg(cams, aff=None, mask=None, cls_labels=None, bkg_score=None):

    b,_,h,w = cams.shape

    bkg = torch.ones(size=(b,1,h,w))*bkg_score
    bkg = bkg.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    cams_with_bkg = torch.cat((bkg, cams), dim=1)

    cams_rw = torch.zeros_like(cams_with_bkg)

    ##########

    b, c, h, w = cams_with_bkg.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask==0] = 0

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-1) ## avoid nan

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _cams = cams_with_bkg[i].reshape(c, -1)
        valid_key = torch.nonzero(cls_labels[i,...])[:,0]
        _cams = _cams[valid_key,...]
        _cams = F.softmax(_cams, dim=0)
        _aff = aff[i]
        _cams_rw = torch.matmul(_cams, _aff)
        cams_rw[i, valid_key,:] = _cams_rw.reshape(-1, cams_rw.shape[2], cams_rw.shape[3])

    return cams_rw


def cams_to_affinity_label(cam_label, mask=None, ignore_index=255):
    
    b,h,w = cam_label.shape
    
    cam_label_resized = F.interpolate(cam_label.unsqueeze(1).type(torch.float32), size=[h//16, w//16], mode="nearest")

    _cam_label = cam_label_resized.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0,2,1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    #aff_label[(_cam_label_rep+_cam_label_rep_t) == 0] = ignore_index
    for i in range(b):

        if mask is not None:
            aff_label[i, mask==0] = ignore_index

        aff_label[i, :, _cam_label_rep[i, 0, :]==ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :]==ignore_index, :] = ignore_index

    return aff_label

def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask