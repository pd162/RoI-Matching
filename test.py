#  Calculate Macro F-score

import torch

from model import Matcher, Matcher_wdq
from dataset import MatchDataset, MatchDataset_CDLA
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict
import torch.nn as nn
from tqdm import tqdm

from metric import SimpleAccuracy

if __name__ == '__main__':
    device = 'cuda'
    model = Matcher_wdq().to(device)
    val_dataset = MatchDataset_CDLA('test_cdla.json', training=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8)

    model_path = '/data1/ljh/code/roi_matching/ckpt/checkpoint-best_7_0.7088204638458023.pth'
    state_dict = torch.load(model_path)

    temp_state = OrderedDict()

    for key, value in state_dict['state_dict'].items():
        temp_state[key.strip('module.')] = value

    model.load_state_dict(temp_state, strict=False)

    model.eval()
    eval_metric = SimpleAccuracy()

    pred_list = []
    gt_list = []
    with torch.no_grad():

        for batch_idx, batch_samples in tqdm(enumerate(val_dataloader)):
            test_img, ref_img, test_mask, ref_mask = batch_samples['test_img'], batch_samples['ref_img'], \
                                                     batch_samples['test_mask'], batch_samples['ref_mask']
            test_img, ref_img, test_mask, ref_mask = test_img.to(device), ref_img.to(device), test_mask.to(
                device), ref_mask.to(device)
            pred = model(ref_img, test_img, ref_mask)
            test_mask = (test_mask != 0).any(dim=1).long()  # 凑合用
            pred = pred.cpu().data.numpy()
            pred = np.argmax(pred, axis=1)

            gt_list.append(test_mask[0].cpu().numpy())
            pred_list.append(pred[0])

    results = eval_metric.process(pred_list, gt_list)
    metrics = eval_metric.compute_metrics(results)
    print(metrics)




