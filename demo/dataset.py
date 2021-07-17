#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : 
@Time    : 2021/5/13 14:49
@File    : dataset.py
@Software: PyCharm
"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataSet
import albumentations as albu

class Dataset(BaseDataSet):
    def __init__(
            self,
            image_dir,
            masks_dir,
            augmentation=None
    ):
        self.ids=os.listdir(image_dir)
        self.images_fps=[os.path.join(image_dir,image_id) for image_id in self.ids]
        self.masks_fps=[os.path.join(masks_dir,image_id) for image_id in self.ids]

        self.augmentation=augmentation

    def __getitem__(self, i):
        #read data
        image=cv2.imread(self.images_fps[i])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask=cv2.imread(self.masks_fps[i],0).astype('int64')

        #apply augmentation
        if self.augmentation:
            sample=self.augmentation(image=image,mask=mask)
            image,mask=sample['image'],sample['mask']

        image=preprocess_input(image).transpose(2,0,1).astype('float32')
        return image,mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentaion():
    train_transform=[
        albu.OneOf([
            albu.HorizontalFlip(p=0.5), #水平翻转
            albu.VerticalFlip(p=0.5),#垂直翻转
            albu.RandomRotate90(p=0.5), #旋转90、180、270
            ]),
        albu.HueSaturationValue(hue_shift_limit=15,sat_shift_limit=15,val_shift_limit=15,p=0.2),
        albu.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.1,rotate_limit=15,p=0.2)
    ]

    return albu.Compose(train_transform)

def preprocess_input(x,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225],input_space="RGB",**kwargs):
    if input_space=="BGR":
        x=x[...,::-1].copy()


    if x.max()>1:
        x=x/255.0

    if mean is not None:
        mean=np.array(mean)
        x=x-mean

    if std is not None:
        std=np.array(std)
        x=x/std

    return x