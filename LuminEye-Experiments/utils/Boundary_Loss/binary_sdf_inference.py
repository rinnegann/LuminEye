"""This file has created for get prediction on the SDF model which has been trained on both ground truth and
SDF Functions"""

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
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torch.load("/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Boundary_Loss.py/Binary_Segemnetation_with_SDF_BY_only_focussing_iris_region_batch_4_epoch_200_boundary_loss/model-0.955.pt")

val_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/val_img"

val_masks =  "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/val_masks/" 
n_classes = 2
batch_size = 1

valid_x = sorted(
        glob(f"{val_images}/*"))
valid_y = sorted(
    glob(f"{val_masks }/*"))



def IoU(pred , true_pred , smooth =1e-10 , n_classes=n_classes):
  with torch.no_grad():
    pred = torch.argmax(F.softmax(pred , dim =1) , dim=1)
    pred = pred.contiguous().view(-1)
    true_pred = true_pred.contiguous().view(-1)

    iou_class = []
    for value in range(0, n_classes):
      true_class = pred == value
      true_label = true_pred == value

      if true_label.long().sum().item()==0:
        iou_class.append(np.nan)
        
      else:
    
        inter = torch.logical_and(true_class, true_label).sum().float().item()
        union = torch.logical_or(true_class , true_label).sum().float().item()

        iou = (inter + smooth)/(union + smooth)
        iou_class.append(iou)

    return np.nanmean(iou_class)

def predict_image_mask(model,image,mask):
    model.eval()
    
    image = image.type(torch.cuda.FloatTensor)
    #image = image.to(device)
    mask = mask.to(device)
    
    # print(f"Original Image shape: {image.size()}")
    
    # print(f"Ground Truth Mask shape: {mask.size()}")
    
    with torch.no_grad():
        
        softmax = nn.Softmax(dim=1)
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        model_output= model(image)
        
        output = softmax(model_output)
        score = IoU(model_output, mask)
        
        masked = torch.argmax(output,dim=1)
        masked = masked.cpu().squeeze(0)
    return masked,score


class Iris(Dataset):
    def __init__(self,images,masks,resize):
        
        self.images = images
        self.masks = masks
        self.resize =resize
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        # print(self.images[index])
        
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        img = img /255.0
        img = cv2.resize(img,self.resize)
        
        
        mask = cv2.imread(self.masks[index],cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,self.resize)
        
        mask = np.where( mask > 0,255,0)
        mask = mask / 255.0
        
        img = torch.from_numpy(img) #(height,width,number_of_channels)
        
        img = img.permute(2,0,1) 
        
        
        msk = torch.from_numpy(mask)
    
        gt_mask = msk[None,:,:]
        
        
    
        
        return img,gt_mask
    
def get_images(test_x,test_y,batch_size=1,shuffle=True,pin_memory=True):
    val_data  = Iris(test_x,test_y,(512,512))
    test_batch = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,drop_last=True)
    return test_batch




val_batch = get_images(valid_x,valid_y,batch_size=batch_size)

val_cls  = Iris(valid_x,valid_y,resize=(512,512))


def main(saved_location,seperate=False):
    
    
    if not os.path.exists(saved_location):
        os.makedirs(saved_location)

    total_iou = 0 
    for i in range(len(val_batch)):
        image,mask = val_cls[i]
        
        pred_mask,iou_score = predict_image_mask(model,image,mask)
        
        image = image.permute(1,2,0)
        image = image.numpy()
        
        
        
        
        mask = mask.permute(1,2,0)
        
        
        gt_mask = mask.numpy()
        
        pred_mask = pred_mask * 255
        pred_mask = np.expand_dims(pred_mask,axis=2)
        
        if seperate:
            if not os.path.exists(os.path.join(saved_location,"img")):
                os.makedirs(os.path.join(saved_location,"img"))
                
            if not os.path.exists(os.path.join(saved_location,"pred")):
                os.makedirs(os.path.join(saved_location,"pred"))
                
            if not os.path.exists(os.path.join(saved_location,"mask")):
                os.makedirs(os.path.join(saved_location,"mask"))
                
            cv2.imwrite(os.path.join(saved_location,"img",f"{i}.png"),image[:,:,::-1]* 255.0)
            cv2.imwrite(os.path.join(saved_location,"pred",f"{i}.png"),pred_mask)
            cv2.imwrite(os.path.join(saved_location,"mask",f"{i}.png"),gt_mask * 255.0)
        else:
            
            line = np.ones((512, 10, 3)) * 128
            
            
            cv2.putText(gt_mask,"GT",(10,10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            
            
            
            cat_images = np.concatenate(
                [img[:,:,::-1], line,pred_mask, line,gt_mask], axis=1
            )
            
            cv2.imwrite(os.path.join(saved_location,f"{i}.png"),cat_images)
        
        total_iou += iou_score
        
    return total_iou



if __name__ == "__main__":
    experiment_name = "experiment_no_1"
    iou = main(experiment_name,seperate=True)
    print(f"Iou Value is {iou/len(val_batch)}")