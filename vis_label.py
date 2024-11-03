import json

# Task 1:  找到kv对 无法使用

# Task 2, 3, 4:  找到关键实体 可以使用
import os.path

import tqdm

train_anno_list = [
    'SVRD/task2/train_label.json',
    # 'SVRD/task3/task3_train/1/label.json',
    # 'SVRD/task3/task3_train/2/label.json',
    # 'SVRD/task3/task3_train/3/label.json',
    # 'SVRD/task3/task3_train/4/label.json',
    # 'SVRD/task3/task3_train/5/label.json',
    # 'SVRD/task3/task3_train/6/label.json',
    # 'SVRD/task3/task3_train/7/label.json',
    # 'SVRD/task3/task3_train/8/label.json',
    # 'SVRD/task3/task3_train/9/label.json',
    # 'SVRD/task3/task3_train/10/label.json',
    # 'SVRD/task3/task3_train/11/label.json',
]
train_img_dir_list = [
    # 'SVRD/task2/train_imgs',
    'SVRD/task3/task3_train/1/label.json',
    'SVRD/task3/task3_train/2/label.json',
    'SVRD/task3/task3_train/3/label.json',
    'SVRD/task3/task3_train/4/label.json',
    'SVRD/task3/task3_train/5/label.json',
    'SVRD/task3/task3_train/6/label.json',
    'SVRD/task3/task3_train/7/label.json',
    'SVRD/task3/task3_train/8/label.json',
    'SVRD/task3/task3_train/9/label.json',
    'SVRD/task3/task3_train/10/label.json',
    'SVRD/task3/task3_train/11/label.json',
]

test_anno_list = [
    'SVRD/task4/task4_train/1/label.json',
    'SVRD/task4/task4_train/2/label.json',
    'SVRD/task4/task4_train/3/label.json',
    'SVRD/task4/task4_train/4/label.json',
    'SVRD/task4/task4_train/5/label.json',
    'SVRD/task4/task4_train/6/label.json',
    'SVRD/task4/task4_train/7/label.json',
    'SVRD/task4/task4_train/8/label.json',
    'SVRD/task4/task4_train/9/label.json',
    'SVRD/task4/task4_train/10/label.json',
    # 'SVRD/task4/task4_train/11/label.json',
]
out_train_anno = 'train_0.json'
out_test_anno = 'test_1.json'


if __name__ == '__main__':

    res_list = []
    for enum_id, train_anno in tqdm.tqdm(enumerate(test_anno_list)):
        with open(train_anno, 'r')as f:
            data = json.load(f)
            f.close()
        valid_entity_list = []
        for entity_info in data['info']['entity']:
            if entity_info[
                'entity_id'
            ] != 0:
                valid_entity_list.append(entity_info['entity_id'])
        for i in range(len(data['image_items']) - 1):
            for j in range(i+1, len(data['image_items'])):
                img_1_info = data['image_items'][i]
                img_2_info = data['image_items'][j]
                for img_1_inst in img_1_info['ocr_instances']:
                    if type(img_1_inst) == dict:
                        img_1_inst = [img_1_inst]
                    # if 'sub_idx' in img_1_inst[0]:
                    #     img_1_entity_id = img_1_inst[0]['sub_idx']
                    # else:
                    img_1_entity_id = img_1_inst[0]['entity_id']
                    if img_1_entity_id == 0:
                        continue
                    for img_2_inst in img_2_info['ocr_instances']:
                        if type(img_2_inst) == dict:
                            img_2_inst = [img_2_inst]
                        # if 'sub_idx' in img_2_inst[0]:
                        #     img_2_entity_id = img_2_inst[0]['sub_idx']
                        # else:
                        img_2_entity_id = img_2_inst[0]['entity_id']
                        if img_2_entity_id == 0:
                            continue
                        if img_1_entity_id == img_2_entity_id:
                            try:
                                res_list.append(
                                dict(
                                    img_1_path=os.path.join(test_anno_list[enum_id].strip('label.json'), img_1_info['image_name']),
                                    img_2_path=os.path.join(test_anno_list[enum_id].strip('label.json'), img_2_info['image_name']),
                                    img_1_inst=img_1_inst[0],
                                    img_2_inst=img_2_inst[0],
                                )
                            )
                            except:
                                continue
        print(train_anno)
    with open(out_test_anno, 'w')as fw:
        json.dump(res_list, fw, indent=4)
        fw.close()
    print(len(res_list))