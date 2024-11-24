import json

from pycocotools.coco import COCO
import os
from tqdm import tqdm
ann_path = '/40/data_center/Document_datasets/DocBank/anno/500K_test.json'
img_dir = '/40/data_center/Document_datasets/DocBank/images'

val_dir = '/40/data_center/Document_datasets/CDLA_DATASET/train'
img_list = []
ann_list = []

for file in os.listdir(val_dir):
    if '.jpg' in file:
        img_list.append(file)
    elif '.json' in file:
        ann_list.append(file)
    else:
        print('invalid')

valid_cats = ['Title', 'Figure', 'Table', 'Equation']

ann_list = ann_list[:1000]
ann_num = len(ann_list)

res_list = []
for ann_id_i in tqdm(range(ann_num - 1)):
    for ann_id_j in range(ann_id_i, ann_num):
        ann_file_path_i = os.path.join(val_dir, ann_list[ann_id_i])
        ann_file_path_j = os.path.join(val_dir, ann_list[ann_id_j])
        img_path_i = ann_file_path_i[:-5] + '.jpg'
        img_path_j = ann_file_path_j[:-5] + '.jpg'

        with open(ann_file_path_i, 'r')as f:
            data_i = json.load(f)
            f.close()
        with open(ann_file_path_j, 'r')as f:
            data_j = json.load(f)
            f.close()

        for inst_i in data_i['shapes']:
            temp_inst_list = []
            if inst_i['label'] in valid_cats:
                for inst_j in data_j['shapes']:
                    if inst_i['label'] == inst_j['label']:
                        temp_inst_list.append(inst_j)
                if len(temp_inst_list) > 0:
                    res_list.append(
                        dict(
                            img_1_path=img_path_i,
                            img_2_path=img_path_j,
                            img_1_inst=inst_i,
                            img_2_inst=temp_inst_list,
                        )
                    )

import random
res_list = random.sample(res_list, 100000)
print(len(res_list))
from collections import Counter
cnt_list = []
for line in res_list:
    cnt_list.append(line['img_1_inst']['label'])
counter = Counter(cnt_list)
print(counter)
with open('train_cdla.json', 'w')as fw:
    json.dump(res_list, fw)
    fw.close()


# valid_cat = [3, 5, 10]
#
#
# if __name__ == '__main__':
#     coco = COCO(ann_path)
#     print(0)
#     res_list = []
#     for img_id_i in tqdm(range(999)):
#         for img_id_j in range(img_id_i + 1, 1000):
#             img_1_info = coco.imgs[img_id_i + 1]
#             img_2_info = coco.imgs[img_id_j + 1]
#             ann_ids_i = coco.getAnnIds([img_id_i + 1])
#             ann_ids_j = coco.getAnnIds([img_id_j + 1])
#             for ann_id_i in ann_ids_i:
#                 img_1_inst = coco.anns[ann_id_i]
#                 temp_inst_list = []
#                 if img_1_inst['category_id'] in valid_cat:
#                     for ann_id_j in ann_ids_j:
#                         img_2_inst = coco.anns[ann_id_j]
#                         if img_1_inst['category_id'] == img_2_inst['category_id']:
#                             # ToDo: 一对多
#                             temp_inst_list.append(img_2_inst)
#
#                     if len(temp_inst_list) > 0:
#                         res_list.append(
#                             dict(
#                                 img_1_path=os.path.join(img_dir, img_1_info['file_name']),
#                                 img_2_path=os.path.join(img_dir, img_2_info['file_name']),
#                                 img_1_inst=img_1_inst,
#                                 img_2_inst=temp_inst_list,
#                             )
#                         )
#
#     import random
#     res_list = random.sample(res_list, 1000)
#     print(len(res_list))
#     from collections import Counter
#     cnt_list = []
#     for line in res_list:
#         cnt_list.append(line['img_1_inst']['category_id'])
#     counter = Counter(cnt_list)
#     print(counter)
#     with open('test_dla.json', 'w')as fw:
#         json.dump(res_list, fw)
#         fw.close()