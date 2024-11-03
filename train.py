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
from losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, LovaszLoss
import torch.distributed as dist
import torch.multiprocessing as mp

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


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--train-root', type=str, default='train_0.json')
    args.add_argument('--val-root', type=str, default='test_1.json')
    args.add_argument('--bs', type=int, default=1)
    args.add_argument('--nw', type=int, default=8)
    args.add_argument('--save-log-dir', type=str, default='./log/')
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--interval', type=int, default=1)
    args.add_argument('--iter-inter', type=int, default=50)
    args.add_argument('--save-ckpt-dir', type=str, default='./ckpt/')

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

def train(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = rank
    model = Matcher().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    train_dataset = MatchDataset(args.train_root, training=True)
    val_dataset = MatchDataset(args.val_root, training=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler,  num_workers=args.nw)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.bs, num_workers=args.nw, shuffle=True)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6,
                                                                     last_epoch=-1)
    DiceLoss_fn = LovaszLoss(mode='multiclass').to(device)
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1).to(device)
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
        train_iter_loss = AverageMeter()
        train_loader_size = train_dataloader.__len__()
        for batch_idx, batch_samples in enumerate(tqdm(train_dataloader)):  # ToDo: 数据维度大小不一致 存在问题
            test_img, ref_img, test_mask, ref_mask = batch_samples['test_img'], batch_samples['ref_img'], batch_samples['test_mask'], batch_samples['ref_mask']
            test_img, ref_img, test_mask, ref_mask = test_img.to(device), ref_img.to(device), test_mask.to(device), ref_mask.to(device)
            pred = model(ref_img, test_img, ref_mask)
            # test_mask = test_mask.to(torch.long)
            test_mask = (test_mask != 0).any(dim=1).long().unsqueeze(0)
            loss = DiceLoss_fn(pred, test_mask) + SoftCrossEntropy_fn(pred, test_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            # train_iter_loss.update(image_loss)

            if batch_idx % args.iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx / train_loader_size * 100,
                    optimizer.param_groups[-1]['lr'],
                    train_epoch_loss.avg, spend_time / (batch_idx + 1) * train_loader_size // 60 - spend_time // 60))
                # train_iter_loss.reset()
        scheduler.step()

        # ToDo: validate the model

        iou = IOUMetric(2)
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(val_dataloader):
                test_img, ref_img, test_mask, ref_mask = batch_samples['test_img'], batch_samples['ref_img'], \
                                                         batch_samples['test_mask'], batch_samples['ref_mask']
                test_img, ref_img, test_mask, ref_mask = test_img.to(device), ref_img.to(device), test_mask.to(
                    device), ref_mask.to(device)
                pred = model(ref_img, test_img, ref_mask)
                test_mask = (test_mask != 0).any(dim=1).long().unsqueeze(0)
                pred = pred.cpu().data.numpy()
                pred = np.argmax(pred, axis=1)
                iou.add_batch(pred, test_mask.cpu().data.numpy())
            acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
            logger.info('[val] epoch:{} iou:{}'.format(epoch, iu))

            if iu[1] > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                filename = os.path.join(args.save_ckpt_dir, 'checkpoint-best.pth')
                torch.save(state, filename)
                best_iou = iu[1]
                # best_mode = copy.deepcopy(model)
                logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))


if __name__ == '__main__':
    args = parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'  # 或者主节点的 IP 地址
    os.environ['MASTER_PORT'] = '29502'
    world_size = torch.cuda.device_count()  # 获取 GPU 数量
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    # train(args)