import logging
import datetime
import numpy as np
from texttable import Texttable
import torch
from PIL import Image

def format_tabs(scores, name_list, cat_list=None):

    _keys = list(scores[0]['iou'].keys())
    _values, _values_precision, _values_recall, _values_confusion_ratio = [],[],[],[]

    for i in range(len(name_list)):
        _values.append(list(scores[i]['iou'].values()))
        _values_precision.append(list(scores[i]['precision'].values()))
        _values_recall.append(list(scores[i]['recall'].values()))
        _values_confusion_ratio.append(list(scores[i]['confusion'].values()))

    _values = np.array(_values) * 100
    _values_precision = np.array(_values_precision) * 100
    _values_recall = np.array(_values_recall) * 100
    _values_confusion_ratio = np.array(_values_confusion_ratio)

    t = Texttable()
    t.header(["Class"] + name_list)

    for i in range(len(_keys)):
        t.add_row([cat_list[i]] + list(_values[:, i]))
    
    t.add_row(["m-Precision"] + list(_values_precision.mean(1)))
    t.add_row(["m-Recall"] + list(_values_recall.mean(1)))
    t.add_row(["m-ConfutionRatio"] + list(_values_confusion_ratio.mean(1)))
    t.add_row(["m-IoU"] + list(_values.mean(1)))

    return t.draw()

def format_tabs_multi_metircs(scores, metric_name, cat_list=None):

    _keys = list(scores[0]['iou'].keys())
    _values = []

    for i, name in enumerate(metric_name):
        _values.append(list(scores[0][name].values()))

    if name != 'confusion':
        _values_all = np.array(_values)
    else:
        _values_all = np.array(_values)

    t = Texttable()
    t.header(["Class"] + metric_name)

    for i in range(len(_keys)):
        t.add_row([cat_list[i]] + list(_values_all[:, i]))

    t.add_row(["average_metrics"] + list(_values_all.mean(1)))

    return t.draw()

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

def resume_train(args,model,optimizer):

    cpt = torch.load(args.resume_train, map_location='cpu')
    begin_iter = cpt['step']
    model_dict = cpt['model_state_dict']
    optim_dict = cpt['optimizer_state_dict']

    new = {}
    for k,v in model_dict.items():
        new[k.replace('module.','')] = v

    model.load_state_dict(new)

    param_groups = model.get_param_groups()
    optim = getattr(optimizer, args.optimizer)(
    params=[
        {
            "params": param_groups[0],
            "lr": args.lr,
            "weight_decay": args.wt_decay,
        },
        {
            "params": param_groups[1],
            "lr": args.lr,
            "weight_decay": args.wt_decay,
        },
        {
            "params": param_groups[2],
            "lr": args.lr * 10,
            "weight_decay": args.wt_decay,
        },
        {
            "params": param_groups[3],
            "lr": args.lr * 10,
            "weight_decay": args.wt_decay,
        },
    ],
    lr=args.lr,
    weight_decay=args.wt_decay,
    betas=args.betas,
    warmup_iter=args.warmup_iters,
    max_iter=args.max_iters,
    warmup_ratio=args.warmup_lr,
    power=args.power)
    
    optim.load_state_dict(optim_dict)
    optim.global_step = begin_iter

    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    
    return model, optim, begin_iter

def save_train(model,optim,ckpt_name,step_,cur_lr):
    torch.save(

        {
            "step": step_,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "lr": cur_lr
        },
        ckpt_name
        )
    
    return True

def convert_test_seg2RGB(out,name):
    arr=out.astype(np.uint8)
    im=Image.fromarray(arr)
 
    palette=[]
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21]=np.array([[0, 0, 0],
                                [128, 0, 0],
                                [0, 128, 0],
                                [128, 128, 0],
                                [0, 0, 128],
                                [128, 0, 128],
                                [0, 128, 128],
                                [128, 128, 128],
                                [64, 0, 0],
                                [192, 0, 0],
                                [64, 128, 0],
                                [192, 128, 0],
                                [64, 0, 128],
                                [192, 0, 128],
                                [64, 128, 128],
                                [192, 128, 128],
                                [0, 64, 0],
                                [128, 64, 0],
                                [0, 192, 0],
                                [128, 192, 0],
                                [0, 64, 128]
                             ], dtype='uint8').flatten()
 
    im.putpalette(palette)
    im.save(name)
