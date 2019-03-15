# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        - root : 数据集目录
        - transforms : 数据增加
        - train / test : 阶段
        """
        # 默认阶段为 train
        self.test = test
        # 组织形式 root/images.jpg
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg 
        if self.test:
            # Return a new list containing all items from the iterable in ascending order.
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 如果是 test 阶段
        if self.test:
            self.imgs = imgs
        # 训练集 分为 70% 训练集，30% 验证集
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        # 对图片的变换（数据增强）
        if transforms is None:
            # 正则化
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 如果是 test 或者 validation
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),  # resize 短边为224
                    T.CenterCrop(224),  # 中心裁剪 224*224
                    T.ToTensor(),  # 将图片转成Tensor，归一化至[0, 1]
                    normalize
                ])
            # train 阶段
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),  # scale=(0.08, 1.0), ratio=(0.75, 1.33) 长宽比
                    T.RandomHorizontalFlip(),  # 水平旋转
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        注意： 将费时操作放在这里，会有多进程加速
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            # dog -> 1 ; cat -> 0
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        # from PIL import Image
        data = Image.open(img_path)
        # 数据增强
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
