# Dataclass Preparation
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
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


IMAGE_DIR = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Iris-Center-Localization/G4_BIO_EYES"
trn_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

RESIZE_AMT = 64


train_transforms =  A.Compose([
    A.Resize(width=RESIZE_AMT,height=RESIZE_AMT),
    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(p=1)
])

val_transforms =  A.Compose([
    A.Resize(width=RESIZE_AMT,height=RESIZE_AMT),
    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(p=1)
])


class CenterDataset(torch.utils.data.Dataset):
    def __init__(self,df,image_dir=IMAGE_DIR,transforms=None):
        self.image_dir = image_dir
        self.df = df
        self.image_ids = df.Image_Name.unique()
        self.transforms = transforms
        
    def __getitem__(self,ix):
        
        img_id = self.image_ids[ix]
        img_path = os.path.join(self.image_dir,img_id)
        
        img = cv2.imread(img_path)[:,:,::-1]
        
        data = self.df[self.df["Image_Name"]==img_id]
        
        
        x1 = data["X1"].values[0] * RESIZE_AMT
        y1 = data["Y1"].values[0] * RESIZE_AMT
        
        center_loc = torch.Tensor([x1,y1]).float()

        if self.transforms:
            transformed = self.transforms(image=img)
            
            image = transformed["image"]
            
    
        return image,center_loc
    def collate_fn(self,batch):
        return tuple(zip(*batch))
    
    def __len__(self):
        return len(self.image_ids)


train_ds = CenterDataset(trn_df,transforms=train_transforms)
test_ds = CenterDataset(val_df,transforms=val_transforms)

trainLoader = DataLoader(train_ds, batch_size=8,
	shuffle=True, num_workers=os.cpu_count(), pin_memory=True,drop_last=True)
testLoader = DataLoader(test_ds, batch_size=8,
	num_workers=os.cpu_count(), pin_memory=True,drop_last=True)


class BB_model(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2  = nn.Sequential(*layers[6:])
    
        self.bb = nn.Sequential(nn.BatchNorm1d(512),nn.Linear(512,2))
        
    def forward(self,x):
        x = self.features1(x) # 1, 128, 32, 32
        
        x = self.features2(x) # [1, 512, 8, 8]
        
        x = F.relu(x)
        
        
        x = nn.AdaptiveAvgPool2d((1,1))(x) # [ 1,512,1,1]
        
        
        x = x.view(x.shape[0],-1) # [1, 512]
        
        return self.bb(x)
    
    
def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr



def main_training(model, optimizer, train_dl, test_dl, epochs):
    idx = 0

    prev_loss = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0

        for x, y_bb in train_dl:
            batch = x.shape[0]
            x = x.cuda().float()

            y_bb = y_bb.cuda()

            out_bb = model(x)

            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()

            loss = loss_bb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx += 1

            total += batch

            sum_loss += loss.item()

        val_loss = val_epochs(model, test_dl)

        if i == 0:
            prev_loss = val_loss
        if val_loss < prev_loss:
            prev_loss = val_loss

            model_name = f"Regression_model_{str(prev_loss)}.pth"
            torch.save(model, model_name)

        train_loss = sum_loss/total

        train_metrics = {"train/epoch": i+1, "train/train_loss": train_loss}

        val_metrics = {"val/epoch": i+1, "val/val_loss": val_loss}
        wandb.log({**train_metrics, **val_metrics})

        print(f"Epoch Number {i+1}")
        print("train_loss %.3f " % (train_loss))
        print("Validation Loss %.3f " % (val_loss))
        print("*"*8)
        
        
def val_epochs(model,val_loader):
    
    model.eval()
    total_val_loss = 0
    total = 0
    for x,y_bb in val_loader:
        
    
        x = x.cuda().float()
        y_bb = y_bb.cuda()
        
        out_bb = model(x)
        
        total += x.shape[0]
        with torch.no_grad():
            loss_bb = F.l1_loss(out_bb,y_bb,reduction='none').sum(1)
            loss_bb = loss_bb.sum()
            
            total_val_loss += loss_bb.item()
            
    return total_val_loss/total

if __name__ == '__main__':
    
    
    config = {"epochs":1000,
                        "max_learning_rate":0.006}


    wandb.init(project="LuminEys-Iris",entity="rinnegann",
            name="Regression_Resent_epoch_1000_l1_loss_batch_8",
            config=config)
    
    model = BB_model().cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.006)

    update_optimizer(optimizer, 0.001)
    main_training(model=model,optimizer=optimizer,train_dl=trainLoader,test_dl=testLoader,epochs=1000)
                