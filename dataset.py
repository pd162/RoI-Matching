import json
import numpy as np
import cv2
import torchvision.transforms
from PIL import Image
import matplotlib
from torch.utils.data import Dataset, DataLoader


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

class MatchDataset(Dataset):
    def __init__(self, root, training=True):
        super(MatchDataset, self).__init__()
        self.root = root
        with open(root, 'r')as f:
            self.data = json.load(f)
            f.close()

        self.training = training
        if training:
            self.img_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize([640, 640]),
                    # torchvision.transforms.RandomCrop(640),
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.RandomVerticalFlip(p=0.5),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ]
            )
            self.label_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize([640, 640]),
                    # torchvision.transforms.RandomCrop(640),
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.RandomVerticalFlip(p=0.5),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                    #                                  std=(0.229, 0.224, 0.225))
                ]
            )
        else:
            self.img_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize([640, 640]),
                    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    # torchvision.transforms.RandomVerticalFlip(p=0.5),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ]
            )
            self.label_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize([640, 640]),
                    # torchvision.transforms.RandomCrop(640),
                    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    # torchvision.transforms.RandomVerticalFlip(p=0.5),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406),
                    #                                  std=(0.229, 0.224, 0.225))
                ]
            )

    def __len__(self):
        return len(self.data)

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))

    def __getitem__(self, item):
        # try:
            inst = self.data[item]
            img_1_path = inst['img_1_path']
            img_2_path = inst['img_2_path']
            img_1_inst = inst['img_1_inst']
            img_2_inst = inst['img_2_inst']

            img1 = Image.open(img_1_path).convert('RGB')
            img2 = Image.open(img_2_path).convert('RGB')

            h1, w1 = img1.size
            h2, w2 = img2.size

            img1_mask = generate_mask(img_1_inst['bbox'], h1, w1)
            img2_mask = generate_mask(img_2_inst['bbox'], h2, w2)

            img1_tensor = self.img_transform(img1)
            img2_tensor = self.img_transform(img2)
            img1_mask_tensor = self.label_transform(img1_mask)
            img2_mask_tensor = self.label_transform(img2_mask)

            res_dict = dict(
                test_img=img1_tensor,
                ref_img=img2_tensor,
                test_mask=img1_mask_tensor,
                ref_mask=img2_mask_tensor
            )
            return res_dict
        # except:
        #     print('Data Error!')
        #     item = self._rand_another()
        #     return self.__getitem__(item)

if __name__ == '__main__':
    dataset = MatchDataset(root='train.json')
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    for data in dataloader:
        print(data['ref_img'].shape)