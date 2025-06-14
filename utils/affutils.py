import torch
import numpy as np
import cv2
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
import torch.nn.functional as F


def compute_trans_mat(attn_weight):
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat

    return trans_mat

def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)

def generate_cam_label(cam_refined_list, w, h):
    refined_cam_to_save = []
    refined_cam_all_scales = []
    for cam_refined in cam_refined_list:
        cam_refined = cam_refined.cpu().numpy().astype(np.float32)
        cam_refined_highres = scale_cam_image([cam_refined], (w, h))[0]
        refined_cam_to_save.append(torch.tensor(cam_refined_highres))

    refined_cam_all_scales.append(torch.stack(refined_cam_to_save,dim=0))

    refined_cam_all_scales = refined_cam_all_scales[0]
    
    return refined_cam_all_scales

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)
    return result

def _refine_cams(ref_mod, images, cams, valid_key,):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())

    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)

def _refine_cams2(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label


def refine_cams_with_bkg_v2(
    ref_mod=None,
    images=None,
    cams=None,
    cls_labels=None,
    high_thre=None,
    low_thre=None,
    ignore_index=False,
    img_box=None,
    down_scale=2,
):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(
        cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False
    )  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(
        cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False
    )  # .softmax(dim=1)

    for idx in range(b):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams2(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams2(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))

        if img_box is not None:
            coord = img_box[idx]
            refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1], coord[2]:coord[3]]
            refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1], coord[2]:coord[3]]
        else:
            refined_label_h[idx] = _refined_label_h[0]
            refined_label_l[idx] = _refined_label_l[0]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def refine_cams_with_bkg_weclip(cam_refined_list, inputs_denorm, cls_lst,par, size):

    w, h = size
    cams = generate_cam_label(cam_refined_list, h, w).cuda()
    bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], 1.).cuda()
    cams = torch.cat([bg_score, cams], dim=0).cuda()

    valid_key = np.pad(cls_lst.numpy()  + 1, (1, 0), mode='constant')
    valid_key = torch.from_numpy(valid_key).cuda()

    with torch.no_grad():
        cam_labels = _refine_cams(par, inputs_denorm, cams, valid_key).unsqueeze(0)

    return cam_labels, cams


def refine_cams_with_aff(attr_map, attn_weights,cls_label, size,caa_thre =0.79, attn_layers=6, seg_attn=None):

    h, w = size
    attn_weight = attn_weights[:, 1:, 1:][-attn_layers:]

    if seg_attn is not None:
        attn_diff = seg_attn - attn_weight
        attn_diff = torch.sum(attn_diff.flatten(1), dim=1)
        diff_th = torch.mean(attn_diff)

        attn_mask = torch.zeros_like(attn_diff)
        attn_mask[attn_diff <= diff_th] = 1
        attn_mask = attn_mask.reshape(-1, 1, 1)
        attn_mask = attn_mask.expand_as(attn_weight)

        ## TODO
        attn_weight = torch.sum(attn_mask*attn_weight, dim=0) / (torch.sum(attn_mask, dim=0) +1e-5)
        attn_weight = attn_weight.detach()
        attn_weight = attn_weight * seg_attn.squeeze(0).detach()
    else:
        attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
        attn_weight = attn_weight.detach()

    _trans_mat = compute_trans_mat(attn_weight)
    _trans_mat = _trans_mat.float()

    cls_lst = torch.where(cls_label)[0]
    cam_refined_list = []

    for i, cls in enumerate(cls_lst):
        grayscale_cam =attr_map[:,cls].cpu().numpy().reshape(h // 16, w // 16)
        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=caa_thre, multi_contour_eval=True)
        aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1])).cuda()
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
        trans_mat = _trans_mat*aff_mask

        cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
        cam_to_refine = cam_to_refine.view(-1, 1)

        cam_refined = torch.matmul(trans_mat , cam_to_refine).reshape(h // 16, w // 16)
        cam_refined_list.append(cam_refined)

    return cam_refined_list, cls_lst.detach().cpu()