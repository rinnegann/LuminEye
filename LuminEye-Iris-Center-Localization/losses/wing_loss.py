import torch
import torch.nn as nn
import numpy as np


class WingLoss(nn.Module):
    def __init__(self, width=5, curvature=0.5):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.C = self.width - self.width * np.log(1 + self.width / self.curvature)

    def forward(self, prediction, target):
        diff = target - prediction
        
        diff_abs = diff.abs()
        
        loss = diff_abs.clone()
        

        idx_smaller = diff_abs < self.width
        idx_bigger = diff_abs >= self.width
        
        
        loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
        loss[idx_bigger]  = loss[idx_bigger] - self.C
        
        
    
        return loss