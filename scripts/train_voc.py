import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append("./")
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
from engine import build_network, build_optimizer, build_validation
from utils.affutils import refine_cams_with_aff, refine_cams_with_bkg_weclip
from model.losses import ( get_seg_loss, get_aff_loss,)
from utils.camutils import cams_to_affinity_label, get_mask_by_radius, lam_to_label, cure_attr_map
from utils import imutils
from utils.PAR import PAR
from torch.utils.tensorboard import SummaryWriter


import warnings 
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

##### Parameter settings
parser.add_argument("--model", default="ExCEL_ViT-B/16", type=str, help="custom clip")
parser.add_argument("--dataset_name", default="pascal_voc", type=str, help="custom clip")
parser.add_argument("--attr_json", default="./attributes_text/descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json", type=str, help="custom clip")
parser.add_argument("--num_attri", default=112, type=int, help="number of attribution tokens")
parser.add_argument("--embedding_dim", default=256, type=int, help="number of attribution tokens")
parser.add_argument("--in_channels", default=768, type=int, help="number of attribution tokens")
parser.add_argument("--radius", default=8, type=int, help="number of attribution tokens")
### loss weight
parser.add_argument("--w_seg", default=1.0, type=float, help="w_seg")
parser.add_argument("--w_diver", default=0.1, type=float, help="w_var")

### training utils
parser.add_argument("--max_iters", default=30000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=50, type=int, help="warmup_iters")

### cam utils
parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75, 1.5), help="multi_scales for cam")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")

### log utils
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--save_ckpt", default=True, action="store_true", help="save_ckpt")
parser.add_argument("--tensorboard", default=False, type=bool, help="log tb")
parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--work_dir", default="w_outputs", type=str, help="w_outputs")
parser.add_argument("--log_tag", default="train_voc", type=str, help="train_voc")

### dataset utils
parser.add_argument("--data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size in training")
parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="train", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")

### optimizer utils
parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=1, type=float, help="poweer factor for poly scheduler")
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
    par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()

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
        # split='train',
        # stage='train',
        split='val',
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

    for n_iter in range(args.max_iters):
        global_step = n_iter + 1
        try:
            name, inputs, cls_labels, img_box, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            name, inputs, cls_labels, img_box, labels= next(train_loader_iter)

        aff_pseudos = []
        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_labels = cls_labels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        segs, fts_diver, attr_maps_raw, attn_weights, attn_pred  = model(inputs)

        if n_iter >= 14000:
            attr_maps_raw = cure_attr_map(model,inputs, ex_feats=fts_diver)
            # attr_maps_raw = cure_attr_map_flip(model, inputs, ex_fts=True, flip=False, raw_fts=fts_diver)

        for i, attr_map in enumerate(attr_maps_raw):
            cls_label = cls_labels[i]
            attn_weight = attn_weights[:,i,...]
            seg_attn = attn_pred[i,...].unsqueeze(0) if n_iter >= 14000 else None
            refined_attr_maps, cls_lst = refine_cams_with_aff(attr_map, attn_weight, cls_label, size=inputs.shape[2:], seg_attn=seg_attn, caa_thre=0.79)
            attr_aff_labels,_ = refine_cams_with_bkg_weclip(refined_attr_maps, inputs_denorm[i], cls_lst, par, size=inputs.shape[2:])
            aff_pseudos.append(attr_aff_labels)
        aff_pseudos = torch.cat(aff_pseudos,dim=0)

        ### seg_loss
        segs = F.interpolate(segs, size=aff_pseudos.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = get_seg_loss(segs, aff_pseudos.type(torch.long), ignore_index=args.ignore_index)
        seg_pred = torch.argmax(segs.detach(),dim=1) 


        mask_size = int(args.crop_size // 16)
        attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
        # aff_mask = cams_to_affinity_label(labels, attn_mask)
        aff_mask = cams_to_affinity_label(seg_pred, mask=attn_mask) if n_iter >= 24000 else cams_to_affinity_label(aff_pseudos, mask=attn_mask)
        _, diver_pseudo_labels = lam_to_label(attr_maps_raw.permute(0,2,1).reshape(4,20,20,20), cls_labels.cuda(), img_box=None, ignore_mid=False, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        diver_loss,_,_ = get_aff_loss(attn_pred, aff_mask)

        # warmup
        loss = 1.0 * seg_loss  + args.w_diver * diver_loss

        avg_meter.add({
            'seg_loss': seg_loss.item(),
            'diver_loss': diver_loss.item(),
        })

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; seg_loss: %.4f, diver_loss: %.4f" % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('diver_loss')))
                if tb_logger is not None:
                    grid_img1, grid_cam1 = make_grid_image(inputs.detach(), attr_maps_raw.detach(), cls_label=cls_labels.detach())
                    # _, grid_cam_aff = make_grid_image(inputs.detach(), attr_maps_aff_all.detach(), cls_label=cls_labels.detach())
                    grid_pseudo_aff = make_grid_label(aff_pseudos.detach())
                    grid_pseudo_mid = make_grid_label(diver_pseudo_labels.detach())
                    grid_gts = make_grid_label(labels.detach())
                    grid_seg_pred = make_grid_label(seg_pred)
                    tb_logger.add_image("visual/img1", grid_img1, global_step=global_step)
                    tb_logger.add_image("visual/cam1", grid_cam1, global_step=global_step)
                    # tb_logger.add_image("visual/cam_aff", grid_cam_aff, global_step=global_step)
                    tb_logger.add_image("visual/pseu_aff", grid_pseudo_aff, global_step=global_step)
                    tb_logger.add_image("visual/pseu_mid", grid_pseudo_mid, global_step=global_step)
                    tb_logger.add_image("visual/seg_gt", grid_gts, global_step=global_step)
                    tb_logger.add_image("visual/seg_pred", grid_seg_pred, global_step=global_step)
    
        
        if (n_iter + 1) % args.eval_iters == 0 and (n_iter + 1) >= 1:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt and (n_iter + 1) >= 2:
                    torch.save(model.state_dict(), ckpt_name)
            tab_results = build_validation(model=model, par=par, val_loader=val_loader)
            if args.local_rank == 0:
                logging.info("\n"+tab_results)

    return True

if __name__ == "__main__":

    args = parser.parse_args()
    timestamp_1 = "{0:%Y-%m}".format(datetime.datetime.now())
    timestamp_2 = "{0:%d-%H-%M-%S}".format(datetime.datetime.now())
    dataset = os.path.basename(args.list_folder)
    exp_tag = f'{dataset}_{args.log_tag}' if 'debug' in args.log_tag  else f'{dataset}_{args.log_tag}_{timestamp_2}'
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