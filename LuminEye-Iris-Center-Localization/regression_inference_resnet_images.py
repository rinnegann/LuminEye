


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

import albumentations as A
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision import models

from torchvision.models.efficientnet import efficientnet_b3
from BaseModels.resnetModels import BB_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from PIL import Image, ImageDraw


regression_model_path = '/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Iris-Center-Localization/BEST_RESNET_REGRESSION_MODEL_FOR_CROPPED_EYES/Regression_model_1.487574208665777.pth'

IMAGE_DIR = "/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD/Images/"

val_df = pd.read_csv("/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD/mix_val.csv")


RESIZE_AMT = 64
BACTH_SIZE = 8


val_transforms =  A.Compose([
    A.Resize(width=RESIZE_AMT,height=RESIZE_AMT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
            
    
        return image,center_loc,img_id
    def collate_fn(self,batch):
        return tuple(zip(*batch))
    
    def __len__(self):
        return len(self.image_ids)



test_ds = CenterDataset(val_df,transforms=val_transforms)


testLoader = DataLoader(test_ds, batch_size=BACTH_SIZE,
	num_workers=os.cpu_count(), pin_memory=True,drop_last=True)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def main(model,test_loader,save_location):
    
    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    image_count = 0
    
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    
    for x,y,img_id in test_loader:
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            out_coord = model(x)
            
        
        
        for i in range(x.shape[0]):
            
            

            image_name = img_id[i]
            
            image = transforms.ToPILImage()(unnorm(x[i]))
            gt_coord = y[i].detach().cpu().numpy()
            pred_coord = out_coord[i].detach().cpu().numpy()
            
            draw = ImageDraw.Draw(image )
            
            
            # gt --> red
            draw.point((int(gt_coord[0]),int(gt_coord[1])), 'red')
            
            
            # prediction --> green
            draw.point((int(pred_coord[0]),int(pred_coord[1])), 'green')
            
            
            image.save(os.path.join(saved_location,image_name))
          
            
        
    
    
if __name__ == '__main__':
    

    saved_location = '/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD/resnet_model_epoch_300'
    regression_model = torch.load(regression_model_path)
    main(regression_model,testLoader,saved_location)