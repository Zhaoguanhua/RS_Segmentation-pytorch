#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : 
@Time    : 2021/5/13 10:54
@File    : train_pointRend.py
@Software: PyCharm
"""

import sys
import torch
from tqdm import tqdm as tqdm
from utils_me.meter import AverageValueMeter
import numpy as np
from utils_me.metrics_myself import Metrics

import torch.nn.functional as F
from segmentation_model.pointrend import point_sample
from utils_me.gpus import reduce_tensor

class Epoch:

    def __init__(self,model,loss,metrics,stage_name,classes,device='cpu',verbose=True):
        self.model=model
        self.loss=loss
        self.metrics=metrics
        self.stage_name=stage_name
        self.classes=classes
        self.verbose=verbose
        self.device=device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self,logs):
        str_logs=['{}-{:.4}'.format(k,v) for k,v in logs.items()]
        s=', '.join(str_logs)
        return s

    def batch_update(self,x,y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self,dataloader):
        self.on_epoch_start()
        logs={}
        loss_meter=AverageValueMeter()
        metrics_meters={metric:AverageValueMeter() for metric in self.metrics}

        all_hist_gpu=Metrics(num_classes=self.classes)
        all_hist_gpu.to(self.device)


        with tqdm(dataloader,desc=self.stage_name,file=sys.stdout,disable=not (self.verbose)) as iterator:
            for x,y in iterator:
                x,y=x.to(self.device),y.to(self.device)

                loss,y_pred = self.batch_update(x,y)

                #update loss logs
                loss_value =loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs={self.loss.__name__:loss_meter.mean}
                logs.update(loss_logs)

                #update metrics logs
                all_hist_gpu.add_batch(y_pred,y)

                if self.stage_name=='train':
                    # calculate acc、iou
                    batch_result=all_hist_gpu.evaluate()

                    metrics_meters_logs={}
                    for metric in self.metrics:
                        metrics_meters[metric].add(batch_result[metric])
                        metrics_meters_logs[metric]=metrics_meters[metric].mean
                    all_hist_gpu.reset()
                    logs.update(metrics_meters_logs)

                if self.verbose:
                    s=self._format_logs(logs)
                    iterator.set_postfix_str(s)

        if self.stage_name=='valid':
            valid_result=all_hist_gpu.evaluate()
            logs.update(valid_result)

        return logs


class TrainEpoch(Epoch):
    def __init__(self,model,loss,metrics,optimizer,classes,lr,max_iter,device='cpu',verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            classes=classes,
            device=device,
            verbose=verbose,
        )
        self.optimizer=optimizer
        self.iter=0
        self.max_iter=max_iter
        self.lr=lr

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self,x,y):
        y=y.squeeze_(1).long()
        self.optimizer.zero_grad()
        result=self.model.forward(x)

        pred=F.interpolate(result["coarse"],x.shape[-2:],mode="bilinear",align_corners=True)
        seg_loss=F.cross_entropy(pred,y,ignore_index=255)

        gt_points=point_sample(
            y.float().unsqueeze(1),
            result["points"],
            mode="nearest",
            align_corners=False
        ).squeeze_(1).long()

        points_loss=F.cross_entropy(result["rend"],gt_points,ignore_index=255)
        loss=seg_loss+points_loss

        reduce_seg=reduce_tensor(seg_loss)
        reduce_point=reduce_tensor(points_loss)
        reduce_loss=reduce_seg+reduce_point

        loss.backward()

        self.optimizer.param_groups[0]['lr']=self.lr*np.power((1-self.iter/self.max_iter),0.9)
        self.iter+=1

        self.optimizer.step()
        return reduce_loss,pred


class ValidEpoch(Epoch):
    def __init__(self,model,loss,metrics,classes,device='cpu',verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            classes=classes,
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self,x,y):
        with torch.no_grad():
            prediction=self.model.forward(x)["fine"]

            loss=F.cross_entropy(prediction,y)

        return loss,prediction





