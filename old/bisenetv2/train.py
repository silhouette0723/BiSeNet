import sys
import os
import os.path as osp
import random
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from bisenetv2.cityscapes_cv2 import get_data_loader
from bisenetv2.evaluatev2 import eval_model
from bisenetv2.ohem_ce_loss import OhemCELoss
from bisenetv2.lr_scheduler import WarmupPolyLrScheduler
from bisenetv2.meters import TimeMeter, AvgMeter
from bisenetv2.logger import setup_logger, print_log_msg
from bisenetv2.hair_dataset import HairSegmentationDataset as Hair_Dt

# 设置要训练的模型文件
MODEL_VARIANTS = ["190", "210", "213", "233"]

d["190"] = 1
d["210"] = 2
d["213"] = 2
d["233"] = 3

# 训练超参数
lr_start = 5e-2
warmup_iters = 1000
max_iter = 150000 + warmup_iters
ims_per_gpu = 8

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', type=int, default=0)
    parse.add_argument('--sync-bn', action='store_true')
    parse.add_argument('--fp16', action='store_true')
    parse.add_argument('--port', type=int, default=44554)
    parse.add_argument('--respth', type=str, default='./bisenetv2/res')
    return parse.parse_args()

args = parse_args()

def set_model(variant):
    module_name = f"bisenetv2.bv2_{variant}MB"
    net_module = __import__(module_name, fromlist=["BiSeNetV2"])
    net = net_module.BiSeNetV2(2)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    num = d[variant]
    criteria_aux = [OhemCELoss(0.7) for _ in range(num)]
    return net, criteria_pre, criteria_aux

def train_model(variant, epochs=100):
    logger = logging.getLogger()
    dataset = Hair_Dt("train")
    dl = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    net, criteria_pre, criteria_aux = set_model(variant)
    optim = torch.optim.SGD(
        [{'params': param, 'weight_decay': 5e-4 if param.dim() in [2, 4] else 0} for param in net.parameters()],
        lr=lr_start, momentum=0.9
    )
    num = d[variant]
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = TimeMeter(max_iter), AvgMeter('loss'), AvgMeter('loss_pre'), [AvgMeter(f'loss_aux{i}') for i in range(num)]
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9, max_iter=max_iter, warmup_iter=warmup_iters, warmup_ratio=0.1, warmup='exp')

    for epoch in range(epochs):
        logger.info(f"Starting training {variant}MB, Epoch {epoch+1}/{epochs}")
        for it, (im, lb) in enumerate(dl):
            im, lb = im.cuda(), lb.cuda().squeeze(1)
            optim.zero_grad()
            logits, *logits_aux = net(im)
            loss = criteria_pre(logits, lb) + sum(crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux))
            loss.backward()
            optim.step()
            lr_schdr.step()

            loss_meter.update(loss.item())
            if (it + 1) % 100 == 0:
                print_log_msg(it, max_iter, sum(lr_schdr.get_lr()) / len(lr_schdr.get_lr()), time_meter, loss_meter, loss_pre_meter, loss_aux_meters)

        save_pth = osp.join(args.respth, f'model_{variant}MB_epoch_{epoch+1}.pth')
        if dist.get_rank() == 0:
            torch.save(net.module.state_dict(), save_pth)
        logger.info(f'Model {variant}MB saved at epoch {epoch+1}')

    eval_model(net, 4)

def main():
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:{args.port}', world_size=torch.cuda.device_count(), rank=args.local_rank)
    os.makedirs(args.respth, exist_ok=True)
    setup_logger('BiSeNetV2-train', args.respth)
    
    for variant in MODEL_VARIANTS:
        train_model(variant)

if __name__ == "__main__":
    main()
