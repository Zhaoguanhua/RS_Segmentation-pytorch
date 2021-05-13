#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : 
@Time    : 2021/5/13 10:33
@File    : metrics_myself.py
@Software: PyCharm
"""
from . import base
# from . import functional as F
# import numpy as np
import torch

class Metrics(base.Metric):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes=num_classes
        self.hist=torch.zeros((num_classes,num_classes))

    def _fast_hist(self,pred,target):

        hist=torch.bincount(self.num_classes*target.int()+pred,
                            minlength=self.num_classes**2).reshape(self.num_classes,self.num_classes)

        return hist

    def add_batch(self,pred,target):

        if pred.shape[1]!=1:
            pred=torch.argmax(pred,dim=1)  #the activation of segmentation head is softmax
        else:
            pred=torch.round(pred).int()   #the activation of segmentation head is sigmoid

        self.hist+=self._fast_hist(pred.flatten(),target.flatten())


    def evaluate(self):
        eps=1e-7
        acc_cls=torch.diag(self.hist).float()/(torch.sum(self.hist,dim=1)+eps)
        mean_acc_cls=torch.mean(acc_cls)
        cls_acc=dict(zip(range(self.num_classes),acc_cls.detach().cpu().numpy()))

        iou=torch.diag(self.hist)/(torch.sum(self.hist,dim=1)+torch.sum(self.hist,dim=0)-torch.diag(self.hist)+eps)
        mean_iou=torch.mean(iou)
        cls_iou=dict(zip(range(self.num_classes),iou.detach().cpu().numpy()))

        return {'Mean Acc':mean_acc_cls.item(),
                'Mean Iou':mean_iou.item(),
                'Class Acc':cls_acc,
                'Class Iou':cls_iou
                }
    def reset(self):
        self.hist.fill_(0)


class Metrics_test():
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes=num_classes

    def add_batch(self,):
        print(self.num_classes)

    def test2(self):
        print("test2")

# a=Metrics_test(num_classes=5)
# a.add_batch()
# a.tes

a=Metrics(num_classes=5)