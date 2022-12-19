import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True,num_classes=2):
        super(TverskyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        true_1_hot = torch.eye(2)[targets.to("cpu").squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.sigmoid(inputs)
        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
            
        
        
        TP = torch.sum(probas * true_1_hot, dims)
        FP = torch.sum((1-targets)*inputs,dim=dims)
        FN = torch.sum(targets*(1-inputs),dim=dims)
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        
        # print(f"Loss FUncation value is {1-Tversky}") 
        
        return 1 - Tversky.mean()
