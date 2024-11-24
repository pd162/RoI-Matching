import json
from typing import OrderedDict

import torch
import cv2
import os
from model import Matcher_wdq
from typing import OrderedDict
from collections import OrderedDict
import numpy as np

if __name__ == '__main__':
    test_json = "test_dla.json"
    device = 'cuda'
    model = Matcher_wdq().to(device)
    ckpt_path = '/data1/ljh/code/roi_matching/ckpt/checkpoint-best_22_0.04839949918199683.pth'
    ckpt = torch.load(ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    with open(test_json, 'r')as f:
        data_list = json.load(f)
        f.close()

    for data in data_list:
        img_1_path = inst['img_1_path']
        img_2_path = inst['img_2_path']
        img_1_inst = inst['img_1_inst']
        img_2_inst = inst['img_2_inst']

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(1, 2, 0).unsqueeze(0)
        # ToDo: 标准化
        pred = model(img_tensor)
        pred = pred.cpu().data.numpy()
        pred = np.argmax(pred, axis=1)
        print(0)