import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models.efficientnet import efficientnet_b3

class BB_model(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = efficientnet_b3(pretrained=True)
        
        layers = list(efficientnet.children())[:1]
        self.features1 = nn.Sequential(*layers)

    
        self.bb = nn.Sequential(nn.BatchNorm1d(1536),nn.Linear(1536,512),nn.ReLU(inplace=True),
                                nn.BatchNorm1d(512),nn.Linear(512,2))
        
    def forward(self,x):
        x = self.features1(x) #[1, 1536, 8, 8]
        x = F.relu(x)
        
        
        x = nn.AdaptiveAvgPool2d((1,1))(x) # [ 1,1536,1,1]
        
        
        x = x.view(x.shape[0],-1) # [1,1536]

        
        
        return self.bb(x)
