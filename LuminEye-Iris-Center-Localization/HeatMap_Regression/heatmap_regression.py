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
import math
import albumentations as A
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision import models
import wandb
import sys
sys.path.append("../BaseModels")
from  unet_model import UNET
device = 'cuda' if torch.cuda.is_available() else 'cpu'



BACTH_SIZE = 32

RESIZE_AMT = 64

IMAGE_DIR = "/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD_mp2gaze/Images"


trn_df = pd.read_csv("/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD_mp2gaze/mix_train.csv")
val_df = pd.read_csv("/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD_mp2gaze/mix_val.csv")





def mean_squared_error(y_true,y_pred):
    """ Return MSE for the Batch"""
    return torch.sum(torch.square(y_pred-y_true),axis=-1).mean()


class CenterDataset(torch.utils.data.Dataset):
    def __init__(self,df,image_dir=IMAGE_DIR,RESIZE_AMT=64):
        
        self.RESIZE_AMT = RESIZE_AMT
        self.image_dir = image_dir
        self.df = df
        self.image_ids = df.Image_Name.unique()
        self.transforms = transforms
        
    # apply gaussian kernel to image
    def _gaussian(self, xL, yL, sigma, H, W):

        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel
    
    
    
    
    # convert original image to heatmap
    def _convertToHM(self, img, keypoints, sigma=2):

        H = img.shape[0] 
        W =  img.shape[1]
        nKeypoints = len(keypoints)

        img_hm = np.zeros(shape=(H, W, nKeypoints // 2), dtype=np.float32)

        for i in range(0, nKeypoints // 2):
            x = keypoints[i * 2]
            y = keypoints[1 + 2 * i]

            channel_hm = self._gaussian(x, y, sigma, H, W)

            img_hm[:, :, i] = channel_hm
            
        return img_hm
    def __getitem__(self,ix):
        
        img_id = self.image_ids[ix]
        img_path = os.path.join(self.image_dir,img_id)
        
        img = cv2.imread(img_path)[:,:,::-1]
        
        
        img = cv2.resize(img,(self.RESIZE_AMT,self.RESIZE_AMT))
        
        img = img/255.0
        
        
        
        data = self.df[self.df["Image_Name"]==img_id]
        
        
        x1 = data["X1"].values[0] * self.RESIZE_AMT
        y1 = data["Y1"].values[0] * self.RESIZE_AMT
        
        
        heatmap = torch.tensor(self._convertToHM(img,[x1,y1]),dtype=torch.float32).permute(2,0,1).view(1*self.RESIZE_AMT*self.RESIZE_AMT)
        
        image = torch.tensor(img,dtype=torch.float32).permute(2,0,1)
        
        
        return image,heatmap
        
    def collate_fn(self,batch):
        return tuple(zip(*batch))
    
    def __len__(self):
        return len(self.image_ids)  


train_ds = CenterDataset(trn_df)
test_ds = CenterDataset(val_df)

trainLoader = DataLoader(train_ds, batch_size=BACTH_SIZE,
	shuffle=True, num_workers=os.cpu_count(), pin_memory=True,drop_last=True)
testLoader = DataLoader(test_ds, batch_size=BACTH_SIZE,
	num_workers=os.cpu_count(), pin_memory=True,drop_last=True)



def trainStep(model,trainLoader,optimizer):
    
    model.train()
    
    epoch_loss = 0
    
    total_step = 0
    
    for _,(x,y) in enumerate(trainLoader):
        
        x = x.to(device)
        y = y.to(device)
        
        
        optimizer.zero_grad()
        
        y_pred = model(x)
        
        loss = mean_squared_error(y,y_pred)
        
        
        loss.backward()
        optimizer.step()
        
        
        epoch_loss += loss.item()
        
        total_step += 1
        
    return epoch_loss/total_step
        
        
def valStep(model,testLoader):
    model.eval()
    
    
    total_val_mse_loss = 0
    total_val_jaccard_index = 0
    
    total_step = 0
    
    for (x,y) in testLoader:
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            
            y_pred = model(x)
        
        #MSE
        loss = mean_squared_error(y,y_pred).item()
        
        total_val_mse_loss += loss
        
        total_step += 1
        
    return total_val_mse_loss/total_step


def main(model,trainLoader,testLoader,optimizer,epochs=100):
    
    
    val_loss = 0
    
    for epoch in range(epochs):
        
        
        
        
        train_epoch_mse_loss=trainStep(model,trainLoader,optimizer)
        val_mse,val_jaccard=valStep(model,testLoader)
        
        
        if epoch==0:
            val_loss = val_mse
            
        elif val_loss<val_mse   and abs(val_loss-val_mse) > 0.2:
            
            model_name = f"hm_model_{str(val_loss)}.pth"
            torch.save(model, model_name)

            
        
        
        
        
        print(f"Epoch {epoch+1}| Train MSE Loss--> {train_epoch_mse_loss}")
        print(f"Epoch {epoch+1}| VAL MSE Loss--> {val_mse}")
        print(f"Epoch {epoch+1}| VAL Jaccard Loss--> {val_jaccard}\n")
        
        
        train_metrics = {"train/epoch": epoch+1, "train/train_MSE_loss": train_epoch_mse_loss}
        val_metrics = {"val/epoch": epoch+1, "val/val_MSE_loss": val_mse,"val/val_Jaccard": val_jaccard}
        wandb.log({**train_metrics, **val_metrics})



if __name__ == "__main__":
    
    
    
    n_epoch = 500
    n_classes = 1
    
    config = {"epochs":n_epoch ,
                        "max_learning_rate":0.006}


    wandb.init(project="LuminEys-Iris",entity="rinnegann",
            name=f"HM_Regression__epoch_{n_epoch}_mse_mean_batch_{BACTH_SIZE}_resize_{RESIZE_AMT}_for_gi4e_bioid_h2head_mp2gaze",
            config=config)
    
    model = UNET(n_classes).to(device)


    parameters = filter(lambda p: p.requires_grad,model.parameters())

    optimizer = torch.optim.Adam(parameters,lr=0.006)
    
    main(model,trainLoader,testLoader,optimizer,epochs=n_epoch)
