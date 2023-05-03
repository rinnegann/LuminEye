import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from glob import glob
import segmentation_models_pytorch as smp
import time 
import wandb
import sys

from boundary_loss_v1 import BoundaryLoss
from boundary_loss import DC_and_BD_loss
from tversky_loss import TverskyLoss,FocalTverskyLoss
from focalLoss import FocalLoss


sys.path.insert(1,"../utils/")
from losses import DiceLoss,IoU,pixel_wise_accuracy,get_lr

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


dice_loss = DiceLoss(mode="multiclass")


def multi_boundaryLossV1_loss_function(y0, y1, y2,y3, y4, y5, y6,y): # Final Argument== Mask
    
    criterion = BoundaryLoss()
    
    loss_1 = criterion(y0,y)
            
    loss_2 =criterion(y1,y)
            
    loss_3 = criterion(y2,y)
             
    loss_4 =criterion(y3,y)
      
    loss_5 = criterion(y4,y)
      
      
    loss_6 = criterion(y5,y)
      
      
    loss_7 = criterion(y6,y)
      
    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
    
    return loss_1,loss

def multi_dice_loss_function(y0, y1, y2,y3, y4, y5, y6,y): # Final Argument== Mask
    loss_1 = dice_loss(y0,y)
            
    loss_2 = dice_loss(y1,y)
            
    loss_3 = dice_loss(y2,y)
             
    loss_4 = dice_loss(y3,y)
      
    loss_5 = dice_loss(y4,y)
      
      
    loss_6 = dice_loss(y5,y)
      
      
    loss_7 = dice_loss(y6,y)
      
    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
    
    return loss_1,loss




def multi_distBinary_loss(y0, y1, y2,y3, y4, y5, y6,y):
    
    criterion = DC_and_BD_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False, 'square': False},{})
    loss_1 = criterion(y0, y)
    loss_2 = criterion(y1, y)
    loss_3 = criterion(y2, y)
    loss_4 = criterion(y3, y)
    loss_5 = criterion(y4, y)
    loss_6 = criterion(y5, y)
    loss_7 = criterion(y6, y)
    
    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
    
    return loss_1,loss

def multi_tversky_loss(y0, y1, y2,y3, y4, y5, y6,y):
    
    criterion = TverskyLoss(num_classes=2)
    loss_1 = criterion(y0, y)
    loss_2 = criterion(y1, y)
    loss_3 = criterion(y2, y)
    loss_4 = criterion(y3, y)
    loss_5 = criterion(y4, y)
    loss_6 = criterion(y5, y)
    loss_7 = criterion(y6, y)
    
    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
    
    return loss_1,loss

def multi_focaltversky_loss(y0, y1, y2,y3, y4, y5, y6,y):
    
    criterion = FocalTverskyLoss()
    loss_1 = criterion(y0, y)
    loss_2 = criterion(y1, y)
    loss_3 = criterion(y2, y)
    loss_4 = criterion(y3, y)
    loss_5 = criterion(y4, y)
    loss_6 = criterion(y5, y)
    loss_7 = criterion(y6, y)
    
    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
    
    return loss_1,loss

def multi_focal_loss(y0, y1, y2,y3, y4, y5, y6,y):
    
    criterion = FocalLoss(gamma=0.7)
    loss_1 = criterion(y0, y)
    loss_2 = criterion(y1, y)
    loss_3 = criterion(y2, y)
    loss_4 = criterion(y3, y)
    loss_5 = criterion(y4, y)
    loss_6 = criterion(y5, y)
    loss_7 = criterion(y6, y)
    
    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7
    
    return loss_1,loss





if __name__ == '__main__':
    pass