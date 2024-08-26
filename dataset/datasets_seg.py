import torch
from torch.utils import data
import os.path as osp
import numpy as np
import random
import cv2

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

from PIL import Image
import os
from torchvision import transforms


class CSTrainValSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(512, 1024), scale=True, mirror=True,
                 ignore_label=-1):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.is_scale = scale
        self.is_mirror = mirror
        self.ignore_label = ignore_label
        # strip()表示删除掉数据中的换行符，for i_id in open(list_path)在图像分割训练中，在list_path对应文件中读取图片的id
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if max_iters:
            # 同时确定最大迭代次数--通过将最大迭代次数除以图片个数后的值的整型与图片个数相乘
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.img_ids = self.img_ids[:max_iters]
        self.files = []
        for item in self.img_ids:
            if len(item) == 2:
                image_path, label_path = item
    # osp.basename返回path最后的文件名，splitext()是用于从后往前切割文件名，【0】是获取最前面的
                name = osp.splitext(osp.basename(label_path))[0]
                img_file = osp.join(self.root, image_path)  # 拼接文件路径
                label_file = osp.join(self.root, label_path)
            #distance_file = osp.join(self.root, distance_path)
                self.files.append({  # append()函数，用于在列表末尾添加新的对象。
                    "img": img_file,
                    "label": label_file,

                    "name": name
                })
            if len(item)>2:
                image_path, label_path, distance_path = item
                # osp.basename返回path最后的文件名，splitext()是用于从后往前切割文件名，【0】是获取最前面的
                name = osp.splitext(osp.basename(label_path))[0]
                img_file = osp.join(self.root, image_path)  # 拼接文件路径
                label_file = osp.join(self.root, label_path)
                # distance_path = item[2]
                distance_file = osp.join(self.root, distance_path)
                self.files.append({  # append()函数，用于在列表末尾添加新的对象。
                    "img": img_file,
                    "label": label_file,
                    "distance": distance_file,
                    "name": name
                })
            # self.id_to_trainid：感觉像是映射表，将两个域的标签统一
        self.id_to_trainid = {0: ignore_label, 1: 0, 2: 1, 3: 2,
                              4: 3, 5: 4, 6: 5}
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.num_class = 6

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 15) / 10.0
        # 宽和高按照一定比例缩放，fx为w方向上的缩放比例
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_LINEAR)  # 双线性插值，在x和y方向根据临近的两个像素的位置进行线性插值
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)  # 最近邻域插值

        # distance = np.load(distance)  # 使用 np.load() 加载 .npy 文件
        # distance = np.resize(distance, (int(distance.shape[0] * f_scale), int(distance.shape[1] * f_scale)))
        return image, label

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy().astype('int32')  # 转换数据类型
        if reverse:
            for v, k in self.id_to_trainid.items():  # item——遍历字典 键-值
                label_copy[label == k] = v  # np数组，将对应位置标签转换成目标域
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]  # 加载数据

        # OpenCV读取图片，cv2.IMREAD_COLOR加载彩色图片
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        # 直接读取成为灰度图片
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        # 加载 distance 图片
        if "distance" in datafiles:
            distance = np.load(datafiles["distance"])
        else:
            distance = None

        label = self.id2trainId(label)

        size = image.shape
        name = datafiles["name"]
        if self.is_scale:
            # image, label, distance = self.generate_scale_label(image, label, distance)
            image, label = self.generate_scale_label(image, label)

        image = np.asarray(image, np.float32)  # 结构数据转化为ndarray，将输入转换为数组
        image = image - np.array([104.00698793, 116.66876762, 122.67891434])

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            # cv2.copyMakeBorder给图片设置边界框
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))

        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        if distance is not None:

            return image.copy(), label.copy(), distance.copy()
        else:
            return image.copy(), label.copy(), name

class CSValSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(512, 1024), ignore_label=-1):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if max_iters:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.img_ids = self.img_ids[:max_iters]
        self.files = []
        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.id_to_trainid = {0: ignore_label, 1: 0, 2: 1, 3: 2,
                              4: 3, 5: 4, 6: 5}
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.num_class = 6

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy().astype('int32')
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        label = self.id2trainId(label)

        size = image.shape
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        image = image - np.array([104.00698793, 116.66876762, 122.67891434])
        img_h, img_w = label.shape
        image = image.transpose((2, 0, 1)).astype(np.float32)

        return image.copy(), label.copy(), name


class CSTestSet(data.Dataset):
    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        self.files = []
        for item in self.img_ids:
            image_path = item[0]
            name = osp.splitext(osp.basename(image_path))[0]
            img_file = osp.join(self.root, image_path)
            self.files.append({
                "img": img_file
            })

        self.num_class = 6

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image = image - np.array([104.00698793, 116.66876762, 122.67891434])
        image = image.transpose((2, 0, 1)).astype(np.float32)
        return image, np.array(size), name