#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : 
@Time    : 2021/5/13 14:49
@File    : train_demo.py
@Software: PyCharm
"""

import os
import sys
sys.path.append("..")
import torch
import datetime

import segmentation_model as smp
import utils_me as utils
import train_pointRend
from dataset import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ENCODER='res50'
CLASSES=['background','building']
DEVICE='cpu'

n_classes=len(CLASSES)

model=smp.PointRend(
    smp.deeplabv3(pretrained=True,resnet=ENCODER,num_classes=n_classes),
    smp.PointHead(in_c=512+n_classes,num_classes=n_classes))
print(model)


root_dir=r"D:\test_data\building_test"

#train data
train_data=os.path.join(root_dir,"train","image")
train_label=os.path.join(root_dir,"train","label")

#test data
valid_data=os.path.join(root_dir,"valid","image")
valid_label=os.path.join(root_dir,"valid","label")

train_dataset=Dataset(train_data,train_label,
                        # augmentation=get_training_augmentation()
                      )
valid_dataset=Dataset(valid_data,valid_label)

batch_size=2
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
valid_loader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0,drop_last=True)


#loss function
loss=utils.losses.CrossEntropyLoss()

train_metrics=['Mean Acc','Mean Iou']
valid_metrics=['Mean Acc','Mean Iou','Class Acc','Class Iou']

loss_name=loss.__name__

base_lr=0.001
max_epoch=100
optimizer=torch.optim.SGD(model.parameters(),lr=base_lr,momentum=0.99,weight_decay=0.0001)

max_iter=round(len(train_dataset)/batch_size)*max_epoch

train_epoch=train_pointRend.TrainEpoch(
    model,
    loss=loss,
    metrics=train_metrics,
    optimizer=optimizer,
    classes=n_classes,
    lr=base_lr,
    max_iter=max_iter,
    device=DEVICE,
    verbose=True
)

valid_epoch=train_pointRend.ValidEpoch(
    model,
    loss=loss,
    metrics=valid_metrics,
    classes=n_classes,
    device=DEVICE,
    verbose=True
)

min_score=99999

#tensorboard
now_time=datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
logs_root = os.path.join("logs",now_time)
train_logs_dir=os.path.join(logs_root,"train")
valid_logs_dir=os.path.join(logs_root,"valid")
train_writer=SummaryWriter(log_dir=train_logs_dir)
valid_writer=SummaryWriter(log_dir=valid_logs_dir)

for i in range(0,max_epoch):
    print('\nEpoch:{}'.format(i))
    print('lr',optimizer.param_groups[0]['lr'])
    train_logs=train_epoch.run(train_loader)
    valid_logs=valid_epoch.run(valid_loader)

    train_writer.add_scalar("lr",optimizer.param_groups[0]['lr'],i)
    train_writer.add_scalar("iou",train_logs['Mean Iou'] , i)
    train_writer.add_scalar("accuracy",train_logs['Mean Acc'], i)
    train_writer.add_scalar("loss",train_logs[loss_name], i)

    valid_writer.add_scalar("iou",valid_logs['Mean Iou'],i)
    valid_writer.add_scalar("accuracy",valid_logs['Mean Acc'], i)
    valid_writer.add_scalar("loss",valid_logs[loss_name], i)

    for class_id in range(n_classes):
        print("iou"+str(class_id),valid_logs["Class Iou"][class_id])
        valid_writer.add_scalar("iou"+str(class_id),valid_logs["Class Iou"][class_id],i)
        valid_writer.add_scalar("accuracy"+str(class_id),valid_logs["Class Acc"][class_id],i)

    for metric in valid_metrics:
        print(metric,valid_logs[metric])

    if min_score>valid_logs[loss_name]:
        min_score=valid_logs[loss_name]
        torch.save(model.state_dict(),ENCODER+'_segmentation_model.pth')
        print('Model saved at epoh {}!'.format(i))

    if i%20==0:
        torch.save(model.state_dict(),ENCODER+"_segmentation_model.pth")

train_writer.close()
valid_writer.close()