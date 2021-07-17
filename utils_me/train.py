import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
from .metrics_myself import Metrics
import numpy as np

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, classes,device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.classes=classes
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        # for metric in self.metrics:
        #     metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric: AverageValueMeter() for metric in self.metrics}


        all_hist_gpu=Metrics(num_classes=self.classes)
        all_hist_gpu.to(self.device)

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                all_hist_gpu.add_batch(y_pred,y)

                if self.stage_name =="train":
                    batch_result=all_hist_gpu.evaluate()
                    metrics_meters_logs={}

                    for metric in self.metrics:
                        metrics_meters[metric].add(batch_result[metric])
                        metrics_meters_logs[metric]=metrics_meters[metric].mean
                    all_hist_gpu.reset()
                    logs.update(metrics_meters_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        if self.stage_name =="valid":
            valid_result=all_hist_gpu.evaluate()
            logs.update(valid_result)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer,classes,lr,max_iter,device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            classes=classes,
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.iter=0
        self.max_iter=max_iter
        self.lr=lr

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()

        #poly learning rate
        self.optimizer.param_groups[0]['lr']=self.lr*np.power((1-self.iter/self.max_iter),0.9)
        self.iter+=1

        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, classes,device='cpu', verbose=True):
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

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
