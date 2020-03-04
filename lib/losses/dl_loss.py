import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import numpy as np


def logplus(x):
    return torch.clamp(
        torch.log(torch.clamp(x, 1e-9)) / np.log(2),  # ~ log_2(max(x, 0))
        0
    )


class DLLoss(_Loss):
    def __init__(self, loss_precision_floor):
        super(DLLoss, self).__init__()
        self.name = "DLLoss"
        self.loss_precision_floor = loss_precision_floor

    def forward(self, pred, target):
        assert not target.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"
        loss = logplus((input - target).abs() / self.loss_precision_floor)
        return loss
