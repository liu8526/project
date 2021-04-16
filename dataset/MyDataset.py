import os
import cv2
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from glob import glob
from torch.utils.data import Dataset

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .transform import train_transform
from .transform import val_transform

class MyDataset(Dataset):
    def __init__(self, imgs_dir1, upsamp = False, transform=None):
        self.transform = transform
        self.ids = [imgs_dir1 + file[:-4] for file in os.listdir(imgs_dir1) if file[-3:] == 'tif']
        print('{} data'.format(len(self.ids)))
        # 上采样
        if upsamp:
            print('upsamp......')
            old_id = copy.deepcopy(self.ids)
            for item in old_id:
                mask = cv2.imread(item + '.png', cv2.IMREAD_GRAYSCALE)
                if ((5 in mask) or(6 in mask) or(7 in mask)):
                    self.ids.append(item)
            print('After upsamp: {} data'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            img_trans = img_nd.transpose(2, 0, 1)
        except:
            print(img_nd.shape)
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = idx + '.tif'
        mask_file = idx + '.png'

        image = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        # image = image.transpose(2, 0, 1)
        
        # image = cv2.imread(img_file[0])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) - 1

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return {
            'image': image / 255,
            'label': mask.long()
        }