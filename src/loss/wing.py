'''
@Author: Jiangtao
@Date: 2019-12-10 18:55:16
@LastEditors: Jiangtao
@LastEditTime: 2020-07-06 14:00:57
@Description: 
'''
from functools import partial
import math
import sys
import os

import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



__all__ = ["WingLoss"]

def wing_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    width: int = 0.5,
    curvature: float = 0.5,
    reduction: str = "sum"
):
    """
    https://arxiv.org/pdf/1711.06753.pdf
    Source https://github.com/BloodAxe/pytorch-toolbelt
    See :class:`~pytorch_toolbelt.losses` for details.
    """
    diff_abs = (targets - outputs).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = \
        width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    c = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - c

    if reduction == "sum":
        loss = loss.sum()
    if reduction == "mean":
        loss = loss.mean()

    return loss


class WingLoss(nn.Module):
    def __init__(
        self,
        width: int = 5,
        curvature: float = 0.5,
        reduction: str = "mean"
    ):
        super().__init__()
        self.loss_fn = partial(
            wing_loss, width=width, curvature=curvature, reduction=reduction
        )

    def forward(self, outputs, targets):
        loss = self.loss_fn(outputs, targets)
        return loss

