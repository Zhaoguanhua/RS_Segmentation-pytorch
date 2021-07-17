import torch.nn as nn
import torch
from . import base
from . import functional as F
from .base import Activation


class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class bce_loss(base.Loss):
    def __init__(self,n_classes,alpha=0.5,smoothing=0.0):
        super(bce_loss,self).__init__()
        self.n_classes=n_classes
        self.alpha=alpha
        self.smoothing=smoothing

    def forward(self,preds,labels):
        eps=1e-7
        #label smoothing
        labels=labels*(1-self.smoothing)+self.smoothing/self.n_classes

        loss_1=-1*self.alpha*torch.log(preds+eps)*labels
        loss_0=-1*(1-self.alpha)*torch.log(1-preds+eps)*(1-labels)

        return torch.mean(loss_0+loss_1)

class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
