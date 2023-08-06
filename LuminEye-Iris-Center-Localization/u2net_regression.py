import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
from torch.utils.data import Dataset
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
import torch.nn.functional as F
import cv2
import time 
import mediapipe
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import wandb
from torchvision.models.feature_extraction import create_feature_extractor
from resnet_regression_training import CenterDataset
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the Model 
model_path = "/home/user/Documents/LuminEye/LuminEye/LuminEye-Experiments/U2net/U2NET_MULTICLASS_IMG_256_DIC_batch_8/Miche_model_2023_04_11_22:14:26_val_iou0.900.pt"
BACTH_SIZE = 8



class BBOX_ENCODER(nn.Module):
    def __init__(self,model_path):
        super().__init__()
        self.model = torch.load(model_path)
        layers = list(self.model.children())[:11]
        self.features = nn.Sequential(*layers)
        


        self.bb = nn.Sequential(nn.BatchNorm1d(512),
                                nn.Linear(512,256),
                                nn.ReLU(),
                                nn.Linear(256,2)
                                )

    def forward(self,x):

        x = self.features(x)
        # x =  F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        
        x = x.view(x.shape[0],-1)

        return self.bb(x)


class BBOX_ENCODER(nn.Module):
    def __init__(self,model_path):
        super().__init__()
        self.model = torch.load(model_path)
        self.return_nodes = ['stage6.rebnconv1d.conv_s1']
        self.feat_ext = create_feature_extractor(self.model,return_nodes=self.return_nodes)
        self.bb = nn.Sequential(nn.BatchNorm1d(512),nn.Linear(512,2))

    def forward(self,x):
        x = self.feat_ext(x)[self.return_nodes[0]]
        x = F.relu(x)

        x = nn.AdaptiveAvgPool2d((1,1))(x)
        
        x = x.view(x.shape[0],-1)

        return self.bb(x)


# class  CenterRegressionModel(nn.Module):
#     def __init__(self,segmentation_path):
#         super().__init__()
#         self.base_model = torch.load(segmentation_path)
        
#         self.numbers = ["1","2","3","4","5","6"]

#         for name,param in self.base_model.named_parameters():
#             if name.split(".")[0][-1] in self.numbers and name.split(".")[0][:-1]=="stage":
                
#                 param.requires_grad_(False)
                
                
#         self.regressor = nn.Sequential(
#                         nn.Flatten(),
#                         nn.Linear(196608,1024),
#                         nn.BatchNorm1d(1024),
#                         nn.Linear(1024,512),
#                         nn.BatchNorm1d(512),
#                         nn.Dropout(0.5),
#                         nn.ReLU(),
#                         nn.Linear(512,256),
#                         nn.Dropout(0.3),
#                         nn.ReLU(),
#                         nn.Linear(256,2)
#                     )
        
#     def forward(self,x):
        
#         d0,d1,d2,d3, d4, d5, d6 =  self.base_model(x)
        
#         # regressor_1 = self.regressor(d1)
        
#         # regressor_2 = self.regressor(d2)
#         # regressor_3 = self.regressor(d3)
        
#         # regressor_4 = self.regressor(d4)
        
#         # regressor_5 = self.regressor(d5)
#         # regressor_6 = self.regressor(d6)
        
#         regressor_main = self.regressor(d0)
        
        
#         return regressor_main


IMAGE_DIR = "/home/user/Documents/LuminEye/LuminEye/LuminEye-Iris-Center-Localization/G4_BIO_EYES"
trn_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

RESIZE_AMT = 64
train_transforms =  A.Compose([
    A.Resize(width=RESIZE_AMT,height=RESIZE_AMT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(p=1)
])

val_transforms =  A.Compose([
    A.Resize(width=RESIZE_AMT,height=RESIZE_AMT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(p=1)
])


    

train_ds = CenterDataset(trn_df,transforms=train_transforms)
test_ds = CenterDataset(val_df,transforms=val_transforms)

# Training 
trainLoader = DataLoader(train_ds, batch_size=BACTH_SIZE,
	shuffle=True, num_workers=os.cpu_count(), pin_memory=True,drop_last=True)
testLoader = DataLoader(test_ds, batch_size=BACTH_SIZE,
	num_workers=os.cpu_count(), pin_memory=True,drop_last=True)




def multi_mse_loss_function(y0,y):
    
    loss_1 = F.l1_loss(y0,y,reduction='none').sum(1).sum()
    # mse_loss = nn.MSELoss(reduction='none')

    # loss_1 = mse_loss(y0,y).sum(1).sum()
        
    return loss_1


def validation(val_dl,model):

    total=0
    epoch_loss = 0

    model.eval()

    for idx,(images,coord) in enumerate(val_dl):
            
            images = images.to(device)
            coord = coord.to(device)
            
            total += images.shape[0]
            
            with torch.no_grad():
                regressor_main= model(images)
                
                
                loss_1= multi_mse_loss_function(regressor_main,
                                                    coord)
            
            
        
            
            epoch_loss += loss_1.item()
    

    return epoch_loss/total

def main_train(train_dl,val_dl,model,epochs):

    prev_loss = 0
    for e in range(epochs):
        model.train()
        total  = 0
        epoch_loss = 0
        for idx,(images,coord) in enumerate(train_dl):
            
            images = images.to(device)
            coord = coord.to(device)
            
            total += images.shape[0]
            
            regressor_main  = model(images)
            
            
            loss = multi_mse_loss_function(regressor_main,
                                                coord)
            
            
        
            
            epoch_loss += loss.item()
            
            loss.backward()
            opt.zero_grad()
            opt.step()    
        
        train_loss = epoch_loss/total

        val_loss = validation(val_dl,model)


        if e == 0:
            prev_loss = val_loss
        if val_loss < prev_loss:
            prev_loss = val_loss

            model_name = f"u2net_regression_{str(prev_loss)}.pth"
            torch.save(model, model_name)

        


        train_metrics = {"train/epoch": e+1, "train/train_loss": train_loss}

        val_metrics = {"val/epoch": e+1, "val/val_loss": val_loss}
        wandb.log({**train_metrics, **val_metrics})

        print(f"Epoch Number {e+1}")
        print("train_loss %.3f " % (train_loss))
        print("Validation Loss %.3f " % (val_loss))
        print("*"*8)

if __name__ == '__main__':

    u2net_model  = BBOX_ENCODER(model_path)
    # u2net_model = CenterRegressionModel(model_path)
    u2net_model.to(device)

    opt = Adam(u2net_model.parameters(),lr=0.0001)

    n_epoch = 30
    
    
    config = {"epochs":n_epoch ,
                        "max_learning_rate":0.0001}


    wandb.init(project="LuminEys-Iris",entity="rinnegann",
            name=f"u2net_regression_epoch_{n_epoch}_mae_summation_batch_{BACTH_SIZE}_resize_{RESIZE_AMT}_u2net_encoder",
            config=config)
    
    
    main_train(train_dl=trainLoader,val_dl=testLoader,model=u2net_model,epochs=n_epoch)
  