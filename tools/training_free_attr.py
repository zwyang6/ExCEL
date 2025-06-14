import clip
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

from datasets import voc
from datasets import clip_text
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.camutils import lam_to_label
from utils import affutils, evaluate, imutils
from utils.PAR import PAR
import imageio.v2 as imageio


import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append("/ExCEL/")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from utils.pyutils import AverageMeter, cal_eta, setup_logger
from utils.tbutils import make_grid_image, make_grid_label
from torch.utils.tensorboard import SummaryWriter
from engine import build_network, build_optimizer, build_validation, build_clip_grad
from utils.affutils import refine_cams_with_aff, refine_cams_with_bkg_weclip

import warnings 
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

##### Parameter settings
parser.add_argument("--model", default="ExCEL_ViT-B/16", type=str, help="custom clip")
parser.add_argument("--dataset_name", default="pascal_voc", type=str, help="custom clip")
parser.add_argument("--attr_json", default="./attributes_text/descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json", type=str, help="custom clip")
parser.add_argument("--num_attri", default=112, type=int, help="number of attribution tokens")
parser.add_argument("--embedding_dim", default=256, type=int, help="number of attribution tokens")
parser.add_argument("--in_channels", default=512, type=int, help="number of attribution tokens")
### loss weight
parser.add_argument("--w_seg", default=0.12, type=float, help="w_seg")
parser.add_argument("--w_dir", default=0.5, type=float, help="w_var")

### training utils
parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")
parser.add_argument("--cam2mask", default=10000, type=int, help="use mask from last layer")

### cam utils
parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--tag_threshold", default=0.2, type=int, help="filter cls tags")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75, 1.5), help="multi_scales for cam")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")

### knowledge extraction
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

### log utils
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--save_ckpt", default=True, action="store_true", help="save_ckpt")
parser.add_argument("--tensorboard", default=True, type=bool, help="log tb")
parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--work_dir", default="w_outputs", type=str, help="w_outputs")
parser.add_argument("--log_tag", default="train_voc", type=str, help="train_voc")

### dataset utils
parser.add_argument("--data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=64, type=int, help="crop_size for local view")
parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--use_aa", type=bool, default=False)
parser.add_argument("--use_gauss", type=bool, default=False)
parser.add_argument("--use_solar", type=bool, default=False)
parser.add_argument("--global_crops_number", type=int, default=2)
parser.add_argument("--local_crops_number", type=int, default=0)

### optimizer utils
parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))
    if args.local_rank == 0 and args.tensorboard == True:
        tb_logger = SummaryWriter(log_dir=args.tb_dir)
    else:
        tb_logger = None

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    device = torch.device(args.local_rank)

    ### build model 
    model, param_groups = build_network(args)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    ### build dataloader 
    train_dataset = voc.VOC12ClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        resize_range=[512, 2048],
        rescale_range=[0.5, 2.0],
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
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

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)
    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    ### build optimizer 
    optim = build_optimizer(args,param_groups)
    logging.info('\nOptimizer: \n%s' % optim)

    # for n_iter in range(args.max_iters):
    #     global_step = n_iter + 1
    #     try:
    #         img_name, inputs, cls_labels, img_box, labels = next(train_loader_iter)
    #     except:
    #         train_loader_iter = iter(train_loader)
    #         img_name, inputs, cls_labels, img_box, labels= next(train_loader_iter)

    aff_pseudo = []
    gts, sms, sms_attr, sms_attr_aff = [], [], [], []

    for _, data in tqdm(enumerate(val_loader),
                        total=len(val_loader), ncols=100, ascii=" >="):
        name, inputs, labels, cls_label = data
        inputs  = F.interpolate(inputs, size=[320, 320], mode='bilinear', align_corners=False)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        segs, attr_maps_raw, attn_weight_list, attn_pred  = model(inputs)

        par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()

        for attr_map in attr_maps_raw:
            refined_attr_maps, cls_lst = refine_cams_with_aff(attr_maps_raw, attn_weight_list, cls_label, size=inputs.shape[2:])
            attr_aff_labels = refine_cams_with_bkg_weclip(refined_attr_maps, inputs[0], cls_lst, par, labels.shape[-1], labels.shape[-2]).unsqueeze(0)
            aff_pseudo.append(attr_aff_labels)
            imageio.imsave('/reference_codes/CLIP_Surgery/results/attr_aff' + "/" + name[0] + ".png", imutils.encode_cmap(np.squeeze(attr_aff_labels.cpu().numpy())).astype(np.uint8))
        # aff_pseudo_labels = torch.stack(aff_pseudo,dim=0)

        sms_attr_aff += list(attr_aff_labels.cpu().numpy().astype(np.int16))
        gts += list(labels.cpu().numpy().astype(np.int16))

    sms_attr_aff_score = evaluate.scores(gts, sms_attr_aff)
    print("sms_attr_aff:")
    print(sms_attr_aff_score)

        # ### seg_loss & reg_loss
        # segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        # seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

        # ### aff loss from ToCo, https://github.com/rulixiang/ToCo
        # clip_pseudo_label = mask_generator(inputs_clip, fmap.clone().detach(), iter_num=n_iter, img_names=img_name, mode='train')
        # pseudo_label =  F.interpolate(clip_pseudo_label.unsqueeze(0).to(torch.float32), size=fmap.shape[2:], mode="nearest").to(torch.int64).squeeze(0)
        # aff_mask = label_to_aff_mask(pseudo_label)
        # ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # # warmup
        # if n_iter <= 2000:
        #     loss = 1.0 * cls_loss  + args.w_ptc * ptc_loss + 0.0 * seg_loss + 0.0 * reg_loss
        # else:
        #     loss = 1.0 * cls_loss  + args.w_ptc * ptc_loss + args.w_seg * seg_loss + args.w_reg * reg_loss

        # cls_pred = (cls > 0).type(torch.int16)
        # cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        # avg_meter.add({
        #     'cls_loss': cls_loss.item(),
        #     'ptc_loss': ptc_loss.item(),
        #     'seg_loss': seg_loss.item(),
        #     'cls_score': cls_score.item(),
        # })

        # optim.zero_grad()
        # loss.backward()
        # optim.step()

        # if (n_iter + 1) % args.log_iters == 0:

        #     delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
        #     cur_lr = optim.param_groups[0]['lr']

        #     if args.local_rank == 0:
        #         logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, ptc_loss: %.4f,  seg_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('ptc_loss'), avg_meter.pop('seg_loss')))
        #         if tb_logger is not None:

        #             grid_img1, grid_cam1 = make_grid_image(inputs.detach(), cams.detach(), cls_label.detach())
        #             grid_seg_gt1 = make_grid_label(refined_pseudo_label.detach())
        #             grid_seg_gt2 = make_grid_label(clip_pseudo_label.detach())
        #             grid_seg_gt3 = make_grid_label(img_lst.detach())
        #             grid_seg_pred = make_grid_label(torch.argmax(segs.detach(), dim=1))
        #             tb_logger.add_image("visual/img1", grid_img1, global_step=global_step)
        #             tb_logger.add_image("visual/cam1", grid_cam1, global_step=global_step)
        #             tb_logger.add_image("visual/CAM_pseu", grid_seg_gt1, global_step=global_step)
        #             tb_logger.add_image("visual/clip_pseu", grid_seg_gt2, global_step=global_step)
        #             tb_logger.add_image("visual/gt", grid_seg_gt3, global_step=global_step)
        #             tb_logger.add_image("visual/seg_pred", grid_seg_pred, global_step=global_step)
        
        # if (n_iter + 1) % args.eval_iters == 0 and (n_iter + 1) >= 1:
        #     ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
        #     if args.local_rank == 0:
        #         logging.info('Validating...')
        #         if args.save_ckpt:
        #             torch.save(model.state_dict(), ckpt_name)
        #     val_cls_score, tab_results = build_validation(model=model, clip=mask_generator, data_loader=val_loader, args=args)
        #     if args.local_rank == 0:
        #         logging.info("val cls score: %.6f" % (val_cls_score))
        #         logging.info("\n"+tab_results)

    return True

if __name__ == "__main__":

    args = parser.parse_args()
    timestamp_1 = "{0:%Y-%m}".format(datetime.datetime.now())
    timestamp_2 = "{0:%d-%H-%M-%S}".format(datetime.datetime.now())
    dataset = os.path.basename(args.list_folder)
    exp_tag = f'{dataset}_{args.log_tag}_{timestamp_2}'
    args.work_dir = os.path.join(args.work_dir, 'voc', timestamp_1, exp_tag)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")
    args.tb_dir = os.path.join(args.work_dir, "tensorboards")


    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)
        os.makedirs(args.tb_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)