#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

# from bisenetv2.bisenetv2 import BiSeNetV2
from bisenetv2.cityscapes_cv2 import get_data_loader
from bisenetv2.evaluatev2 import eval_model
from bisenetv2.ohem_ce_loss import OhemCELoss
from bisenetv2.lr_scheduler import WarmupPolyLrScheduler
from bisenetv2.meters import TimeMeter, AvgMeter
from bisenetv2.logger import setup_logger, print_log_msg

from bisenetv2.hair_dataset import HairSegmentationDataset as Hair_Dt
from torch.utils.data import DataLoader

# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False


## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')

lr_start = 5e-2
warmup_iters = 3000
max_iter = 150000  + warmup_iters
ims_per_gpu = 8

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,gamma=2,weight=None):
        '''
        alpha: positive samples ratio
        '''
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if weight is None:
            self.weight = None
        else:
            weight = np.array(weight, dtype=np.float32)
            self.weight = torch.from_numpy(weight).reshape((-1,)).cuda()
    def forward(self, inputs, targets):
        '''
        inputs: BxCxHxW
        targets: BxCxHxW
        '''
        ce_loss = F.cross_entropy(inputs, targets, weight= self.weight, reduction = 'none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma*ce_loss
        return focal_loss.mean()

d = {}
d["170"] = 0
d["190"] = 1
d["210"] = 2
d["214"] = 2
d["233"] = 3

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=0,)
    parse.add_argument('--sync-bn', dest='use_sync_bn', action='store_true',)
    parse.add_argument('--fp16', dest='use_fp16', action='store_true',)
    parse.add_argument('--port', dest='port', type=int, default=44554,)
    parse.add_argument('--respth', dest='respth', type=str, default='./bisenetv2/res',)
    parse.add_argument('--loadmodel_size', dest='loadmodel_size', type=int)
    return parse.parse_args()

args = parse_args()

def set_model():
    varient = args.loadmodel_size
    model_name = f"bisenetv2.bv2_{varient}MB"
    net_module = __import__(model_name, fromlist=["bisenetv2"])
    net = net_module.BiSeNetV2(2)
    if args.use_sync_bn: net = set_syncbn(net)
    net.cuda()
    net.train()
    num = d[f'{varient}']
    criteria_pre = FocalLoss(gamma=2)
    criteria_aux = [FocalLoss(gamma=2) for _ in range(num)]
    return net, criteria_pre, criteria_aux

def set_syncbn(net):
    if has_apex:
        net = parallel.convert_syncbn_model(net)
    else:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def set_optimizer(model):
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if param.dim() == 1:
            non_wd_params.append(param)
        elif param.dim() == 2 or param.dim() == 4:
            wd_params.append(param)
    params_list = [
        {'params': wd_params, },
        {'params': non_wd_params, 'weight_decay': 0},
    ]
    optim = torch.optim.SGD(
        params_list,
        lr=lr_start,
        momentum=0.9,
        weight_decay=5e-4,
    )
    return optim


def set_model_dist(net):
    if has_apex:
        net = parallel.DistributedDataParallel(net, delay_allreduce=True)
    else:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank,
            find_unused_parameters=True)
    return net


def set_meters():
    time_meter = TimeMeter(max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    num = d[f"{args.loadmodel_size}"]
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i)) for i in range(num)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def save_model(states, save_pth):
    logger = logging.getLogger()
    logger.info('\nsave models to {}'.format(save_pth))
    for name, state in states.items():
        save_name = 'model_final_{}.pth'.format(name)
        modelpth = osp.join(save_pth, save_name)
        if dist.is_initialized() and dist.get_rank() == 0:
            torch.save(state, modelpth)


def train(epochs=100):  # 允许传入 epochs 参数，默认为 100 轮
    logger = logging.getLogger()
    is_dist = dist.is_initialized()

    dataset = Hair_Dt("train")
    dl = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    net, criteria_pre, criteria_aux = set_model()
    optim = set_optimizer(net)

    if has_apex and args.use_fp16:
        net, optim = amp.initialize(net, optim, opt_level='O1')

    net = set_model_dist(net)

    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    lr_schdr = WarmupPolyLrScheduler(
        optim, power=0.9, max_iter=max_iter, warmup_iter=warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,
    )

    for epoch in range(epochs):  # 训练多个 epoch
        logger.info(f"Starting epoch {epoch+1}/{epochs}...")

        for it, (im, lb) in enumerate(dl):
            im = im.cuda()
            lb = lb.cuda()
            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            logits, *logits_aux = net(im)
            loss_pre = criteria_pre(logits, lb)
            loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)

            if has_apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()
            torch.cuda.synchronize()
            lr_schdr.step()

            time_meter.update()
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

            if (it + 1) % 100 == 0:
                lr = lr_schdr.get_lr()
                lr = sum(lr) / len(lr)
                print_log_msg(it, max_iter, lr, time_meter, loss_meter, loss_pre_meter, loss_aux_meters)

        # 每个 epoch 结束后保存模型
        if (epoch + 1) % 25 == 0:  # 例如，每 10 个 epoch 保存一次
#         if True:
            save_pth = osp.join(args.respth, f'model_{args.loadmodel_size}MB_epoch_{epoch+1}.pth')
            if dist.get_rank() == 0:
                torch.save(net.module.state_dict(), save_pth)
            logger.info(f'Model saved at epoch {epoch+1}')

    logger.info('\nFinal model saved')

    logger.info('\nEvaluating the final model...')
    torch.cuda.empty_cache()
    eval_model(net, 4)

    return


def main():
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    if not osp.exists(args.respth): os.makedirs(args.respth)
    setup_logger('BiSeNetV2-train', args.respth)
    train()


if __name__ == "__main__":
    main()