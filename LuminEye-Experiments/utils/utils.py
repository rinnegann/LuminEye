import torch
import os
import numpy as np
import shutil
from rich.progress import track


def createTrainValSplit(dataset_path,img_folder,mask_folder,ext,split_amt=0.9):
    
    
    full_dataset = os.listdir(os.path.join(dataset_path,img_folder))
    train_size = int(split_amt* len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    
    for train_i,val_i in track(zip(list(train_dataset),list(test_dataset)),total=len(list(train_dataset)),description='Data Splitting'):
    
        img_path = os.path.join(dataset_path,img_folder,val_i)
        masks_path = os.path.join(dataset_path,mask_folder,val_i.split(".")[0]+"."+ext)
        
        move_files(img_path,masks_path,os.path.join(dataset_path,"val_img"),os.path.join(dataset_path,"val_masks"))
        
        
def move_files(img_name,mask_name,img_target,mask_target):
    [os.makedirs(x) for x in [img_target,mask_target] if not os.path.exists(x)]
    
    shutil.move(img_name,img_target)
    shutil.move(mask_name,mask_target)
    
    
if __name__ == "__main__":
    DATA_DIR = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/"
    createTrainValSplit(DATA_DIR,"img", "masks","bmp")