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
# device = "cpu"

RESIZE_AMT = 64


class AdaptiveWingLoss(nn.Module):
  def __init__(self, alpha=2.1, omega=14.0, theta=0.5, epsilon=1.0,\
               whetherWeighted=False, dilaStru=3, w=10, device=device):
    super(AdaptiveWingLoss, self).__init__()
    self.device = device
    self.alpha = torch.Tensor([alpha]).to(device)
    self.omega = torch.Tensor([omega]).to(device)
    self.theta = torch.Tensor([theta]).to(device)
    self.epsilon = torch.Tensor([epsilon]).to(device)
    self.dilationStru = dilaStru
    self.w = torch.Tensor([w]).to(device)
    self.tmp = torch.Tensor([self.theta / self.epsilon]).to(device)
    self.wetherWeighted = whetherWeighted

# '''
#    #param predictions: predicted heat map with dimension of batchSize * landmarkNum * heatMapSize * heatMapSize  
#    #param targets: ground truth heat map with dimension of batchSize * landmarkNum * heatMapSize * heatMapSize  
# '''
  def forward(self, predictions, targets):
    
    
    deltaY = predictions - targets
    deltaY = torch.abs(deltaY)
    alphaMinusY = self.alpha - targets
    
    
    a = self.omega / self.epsilon * alphaMinusY / (1 + self.tmp.pow(alphaMinusY))\
        * self.tmp.pow(alphaMinusY - 1)
    c = self.theta * a - self.omega * torch.log(1 + self.tmp.pow(alphaMinusY))

    l = torch.where(deltaY < self.theta,
                    self.omega * torch.log(1 + (deltaY / self.epsilon).pow(alphaMinusY)),
                    a * deltaY - c)
    if self.wetherWeighted:
      weightMap = self.grayDilation(targets, self.dilationStru)
      weightMap = torch.where(weightMap >= 0.2, torch.Tensor([1]).to(self.device),\
                              torch.Tensor([0]).to(self.device))
      l = l * (self.w * weightMap + 1)

    l = torch.mean(l)

    return l
    
  def grayDilation(self, heatmapGt, structureSize):
    batchSize, landmarkNum, heatmapSize, _ = heatmapGt.shape
    weightMap = heatmapGt.clone()
    step = structureSize // 2
    for i in range(1, heatmapSize-1, 1):
      for j in range(1, heatmapSize-1, 1):
        weightMap[:, :, i, j] = torch.max(heatmapGt[:, :, i - step: i + step + 1,\
                                j - step: j + step + 1].contiguous().view(batchSize,\
                                landmarkNum, structureSize * structureSize), dim=2)[0]

    return weightMap


IMAGE_DIR = "/home/nipun/Documents/Uni_Malta/Datasets/CenterRegression/MixDataset/images"
trn_df = pd.read_csv("/home/nipun/Documents/Uni_Malta/Datasets/CenterRegression/MixDataset/trainAll.csv")
val_df = pd.read_csv("/home/nipun/Documents/Uni_Malta/Datasets/CenterRegression/MixDataset/valAll.csv")


class CenterDataset(torch.utils.data.Dataset):
    def __init__(self,df,image_dir=IMAGE_DIR,RESIZE_AMT=RESIZE_AMT):
        
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
        
        
        heatmap = torch.tensor(self._convertToHM(img,[x1,y1]),dtype=torch.float32).permute(2,0,1)#.view(1*self.RESIZE_AMT*self.RESIZE_AMT)
        
        image = torch.tensor(img,dtype=torch.float32).permute(2,0,1)
        
        
        return image,heatmap
        
    def collate_fn(self,batch):
        return tuple(zip(*batch))
    
    def __len__(self):
        return len(self.image_ids) 
    
    
BACTH_SIZE = 128
train_ds = CenterDataset(trn_df)
test_ds = CenterDataset(val_df)

trainLoader = DataLoader(train_ds, batch_size=BACTH_SIZE,
	shuffle=True, num_workers=os.cpu_count(), pin_memory=True,drop_last=True)
testLoader = DataLoader(test_ds, batch_size=BACTH_SIZE,
	num_workers=os.cpu_count(), pin_memory=True,drop_last=True)




def double_conv(input_channels,output_channels):
    return nn.Sequential(nn.Conv2d(input_channels,output_channels,kernel_size=3,stride=1,padding='same'),
                         nn.BatchNorm2d(output_channels),
                         nn.ReLU(),
                         nn.Conv2d(output_channels,output_channels,kernel_size=3,stride=1,padding='same'),
                         nn.BatchNorm2d(output_channels),
                         nn.ReLU())
    
    
class UNET(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        
        self.num_classes = num_classes
        self.conv_1 = double_conv(3,64)
        self.mx_1 = nn.MaxPool2d(kernel_size=(2,2))
        
        
        self.conv_2 = double_conv(64,128)
        self.mx_2 = nn.MaxPool2d(kernel_size=(2,2))
        
        
        self.conv_3 = double_conv(128,256)
        self.mx_3 = nn.MaxPool2d(kernel_size=(2,2))
        
        
        
        self.conv_4 = double_conv(256,512)
        self.mx_4 = nn.MaxPool2d(kernel_size=(2,2))
        
        
        
        self.conv_5 = double_conv(512,1024)
        
        
        
        self.up_1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        
        
        
        self.conv6 = double_conv(1024,512)
       
       
        self.up_2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        
        
        
        self.conv7 = double_conv(512,256)
        
        
        self.up_3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        
        
        self.conv8 = double_conv(256,128)
        

        self.up_4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        
        self.conv9 = double_conv(128,64)
        
        self.out_conv = nn.Conv2d(64,self.num_classes,kernel_size=1)
        
    def forward(self,x):

        
        x1 = self.conv_1(x)  # 1, 64, 64,64
        
        x1_max = self.mx_1(x1)
        
        #################
        
        x2 = self.conv_2(x1_max)  # 1, 128, 32, 32
        x2_max = self.mx_2(x2)
        #################
        x3 = self.conv_3(x2_max)  # 1, 256, 16,16
        x3_max = self.mx_3(x3)
        ################
        
        x4 = self.conv_4(x3_max) # 1, 512, 8, 8
        x4_max = self.mx_4(x4)
        
        #################
        
        # Bridge
        
        x5 = self.conv_5(x4_max)  # 1, 1024, 4, 4
       
        ####################
        
        x6 = self.up_1(x5) # [1, 512, 8, 8]
      
        
        
        decoder_1 = torch.concat([x6,x4],dim=1) # 1, 1024, 8, 8
        
        
        decoder_1 = self.conv6(decoder_1)  # [1, 512, 8, 8]
        
        
        
        
        d_2 = self.up_2(decoder_1)
        
        decoder_2 = torch.concat([d_2,x3],dim=1) # [1, 512, 16, 16]
        
        decoder_2 = self.conv7(decoder_2) # [1, 256, 16, 16]
        
       
        d_3 = self.up_3(decoder_2)
        
        decoder_3 = torch.concat([d_3,x2],dim=1)
        
        decoder_3 = self.conv8(decoder_3) # [1, 128, 32, 32]
        
        
        d_4 = self.up_4(decoder_3)
        decoder_4 = torch.concat([d_4,x1],dim=1)
        
        decoder_4 = self.conv9(decoder_4) # 1, 64, 64, 64
        
        output = self.out_conv(decoder_4)
        
        return output
    
    
criterion = AdaptiveWingLoss(whetherWeighted=True)


def trainStep(model,trainLoader,optimizer,loss_fn):
    
    model.train()
    
    epoch_loss = 0
    
    total_step = 0
    
    for _,(x,y) in enumerate(trainLoader):
        
        x = x.to(device)
        y = y.to(device)
        
        
        optimizer.zero_grad()
        
        y_pred = model(x)
        
        loss = loss_fn(y_pred,y)
        
        
        loss.backward()
        optimizer.step()
        
        
        epoch_loss += loss.item()
        
        total_step += 1
        
    return epoch_loss/total_step

def valStep(model,testLoader,loss_fn):
    model.eval()
    
    
    total_val_loss = 0
    
    total_step = 0
    
    for (x,y) in testLoader:
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            
            y_pred = model(x)
        
    
        loss = loss_fn(y_pred,y).item()
        
        total_val_loss  += loss
        
        total_step +=1
        
    return total_val_loss /total_step


def main(model,trainLoader,testLoader,optimizer,loss_fn,epochs=100):
    
    
    val_loss = 0
    
    for epoch in range(epochs):
        
        
        
        
        trainLoss=trainStep(model,trainLoader,optimizer,loss_fn)
        valLoss=valStep(model,testLoader,loss_fn)
        
        
        if epoch==0:
            val_loss = valLoss
            
        elif valLoss<val_loss  and abs(val_loss-valLoss) > 0.1:
            
            model_name = f"hm_model_{str(val_loss)}.pth"
            torch.save(model, model_name)

            
        
        
        
        
        print(f"Epoch {epoch+1}| Train Adaptive Loss--> {trainLoss}")
        print(f"Epoch {epoch+1}| VAL Adaptive Loss--> {valLoss}")
    
        
        
if __name__ =="__main__":
    model = UNET(num_classes=1).to(device)
    parameters = filter(lambda p: p.requires_grad,model.parameters())

    optimizer = torch.optim.AdamW(parameters,lr=1e-3)

    main(model,trainLoader,testLoader,optimizer,criterion,epochs=100)