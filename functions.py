import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CosineLoss(nn.Module):
    def __init__(self, num_classes, m=0.35, s=64):
        super(CosineLoss, self).__init__()
        self.num_classes = num_classes
        self.m = m
        self.s = s

    def forward(self, xw_norm, labels):
        label_one_hot = F.one_hot(labels.view(-1), self.num_classes).float() * self.m
        value = self.s * (xw_norm - label_one_hot)
        return F.cross_entropy(input=value, target=labels.view(-1))


class Linear(nn.Module):
    def __init__(self, num_classes):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(512, num_classes))

    def forward(self, x):
        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        w_norm = self.w / torch.norm(self.w, dim=0, keepdim=True)
        xw_norm = torch.matmul(x_norm, w_norm)
        return xw_norm