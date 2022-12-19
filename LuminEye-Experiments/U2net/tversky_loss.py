import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True,num_classes=2):
        super(TverskyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        
        TP = 0
        FP = 0
        FN = 0
         
        targets = targets.view(-1)
        
        for index in range(self.num_classes):
            TP += (inputs[:,index,:,:].view(-1)* targets).sum()
            FP += ((1-targets)*inputs[:,index,:,:].view(-1)).sum()
            FN += (targets * (1-inputs[:,index,:,:].view(-1))).sum()
        
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky