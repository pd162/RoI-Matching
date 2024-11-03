import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_mask(coords, width, height):
    """
    生成一个二值化的mask.

    参数:
    coords: list, 包含八个坐标 [x1, y1, x2, y2, x3, y3, x4, y4]
    width: int, 图像的宽度
    height: int, 图像的高度

    返回:
    numpy.ndarray, 二值化mask
    """
    # 创建一个全零的mask
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # 将坐标转换为NumPy数组并重塑为适合cv2.fillPoly的形状
    points = np.array([[coords[0], coords[1]],
                       [coords[2], coords[3]],
                       [coords[4], coords[5]],
                       [coords[6], coords[7]]], dtype=np.int32)

    # 填充多边形区域
    cv2.fillPoly(mask, [points], (255, 255, 255))
    return Image.fromarray(mask)

if __name__ == '__main__':
    data_path = 'train_0.json'
    with open(data_path, 'r')as f:
        data = json.load(f)

    for inst in data:
        img_1_path = inst['img_1_path']
        img_2_path = inst['img_2_path']
        img_1_inst = inst['img_1_inst']['bbox']
        img_2_inst = inst['img_2_inst']['bbox']

        img1 = Image.open(img_1_path)
        img2 = Image.open(img_2_path)

        w1, h1 = img1.size
        w2, h2 = img2.size

        img1_mask = generate_mask(img_1_inst, w1, h1)
        img2_mask = generate_mask(img_2_inst, w2, h2)

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0, 0].imshow(img1)
        axs[0, 1].imshow(img1_mask)
        axs[1, 0].imshow(img2)
        axs[1, 1].imshow(img2_mask)
        plt.show()
        print(0)


