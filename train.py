import argparse
import logging
import os
import time
import torch
import numpy as np
import copy
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Matcher
from dataset import MatchDataset
import torch.optim as optim
from losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, LovaszLoss, OHEMCrossEntropyLoss
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import math

class IOUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--train-root', type=str, default='test_1.json')
    args.add_argument('--val-root', type=str, default='test_1.json')
    args.add_argument('--bs', type=int, default=16)
    args.add_argument('--nw', type=int, default=8)
    args.add_argument('--save-log-dir', type=str, default='./log/')
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--interval', type=int, default=1)
    args.add_argument('--iter-inter', type=int, default=50)
    args.add_argument('--save-ckpt-dir', type=str, default='./ckpt/')
    args.add_argument('--resume', type=str, default='/data1/ljh/code/roi_matching/ckpt/checkpoint-best_36_0.7941891740814542.pth')
    args.add_argument('--local_rank', type=int, help="local gpu id", default=-1)
    args.add_argument('--lr', type=float, default=1e-3)

    return args.parse_args()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def train(args):
    local_rank = args.local_rank
    dist.init_process_group("nccl", init_method='env://')
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    device = local_rank
    model = Matcher().cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2,
    #                                                            eta_min=1e-6,
    #                                                            last_epoch=-1)  # 并不是最优策略 基本前10个epoch有点浪费

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                find_unused_parameters=True)

    if len(args.resume) > 0:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    scaler = GradScaler()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    train_dataset = MatchDataset(args.train_root, training=True)
    val_dataset = MatchDataset(args.val_root, training=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler,  num_workers=args.nw)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, num_workers=args.nw)

    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, wei
    # ght_decay=1e-4)

    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 5, eta_min=1e-6, last_epoch=-1)
    # DiceLoss_fn = LovaszLoss(mode='multiclass').cuda()
    DiceLoss_fn = DiceLoss(mode='multiclass')
    Lovas_fn = LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
    # SoftCrossEntropy_fn = OHEMCrossEntropyLoss()
    if not os.path.exists(args.save_log_dir):
        os.mkdir(args.save_log_dir)
    if not os.path.exists(args.save_ckpt_dir):
        os.mkdir(args.save_ckpt_dir)
    logger = get_logger(
        os.path.join(args.save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) + '_' + 'matching' + '.log'))
    # scaler = GradScaler()
    best_iou = 0.
    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = AverageMeter()
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        train_loader_size = train_dataloader.__len__()
        for batch_idx, batch_samples in enumerate(tqdm(train_dataloader)):
            train_iter_loss = AverageMeter()
            test_img, ref_img, test_mask, ref_mask = batch_samples['test_img'], batch_samples['ref_img'], batch_samples['test_mask'], batch_samples['ref_mask']
            test_img, ref_img, test_mask, ref_mask = test_img.cuda(), ref_img.cuda(), test_mask.cuda(), ref_mask.cuda()

            with autocast():
                pred = model(ref_img, test_img, ref_mask)
                # test_mask = test_mask.to(torch.long)
                test_mask = (test_mask != 0).any(dim=1).long()
                # loss = SoftCrossEntropy_fn(pred, test_mask) + Lovas_fn(pred, test_mask)  # ToDo: OHEM BCELoss
                loss = DiceLoss_fn(pred, test_mask)

                scaler.scale(loss).backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # optimizer.step()

            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)

            if batch_idx % args.iter_inter == 0 and global_rank == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx / train_loader_size * 100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg, spend_time / (batch_idx + 1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()
        # train_epoch_loss.update(image_loss)
        scheduler.step()
        # scheduler.step()
        iou = IOUMetric(2)
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(val_dataloader):
                test_img, ref_img, test_mask, ref_mask = batch_samples['test_img'], batch_samples['ref_img'], \
                                                         batch_samples['test_mask'], batch_samples['ref_mask']
                test_img, ref_img, test_mask, ref_mask = test_img.cuda(), ref_img.cuda(), test_mask.to(
                    device), ref_mask.cuda()
                pred = model(ref_img, test_img, ref_mask)
                test_mask = (test_mask != 0).any(dim=1).long()  # 凑合用
                pred = pred.cpu().data.numpy()
                pred = np.argmax(pred, axis=1)
                iou.add_batch(pred, test_mask.cpu().data.numpy())
            acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
            if global_rank == 0:
                logger.info('[val] epoch:{} iou:{}'.format(epoch, iu))

            if iu[1] > best_iou and global_rank == 0:  # train_loss_per_epoch valid_loss_per_epoch
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                filename = os.path.join(args.save_ckpt_dir, f'checkpoint-best_{epoch}_{iu[1]}.pth')
                torch.save(state, filename)
                best_iou = iu[1]
                # best_mode = copy.deepcopy(model)
                logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))


if __name__ == '__main__':
    args = parse_args()
    # os.environ['MASTER_ADDR'] = 'localhost'  # 或者主节点的 IP 地址
    # os.environ['MASTER_PORT'] = '29506'
    # world_size = torch.cuda.device_count()  # 获取 GPU 数量
    # mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    train(args)