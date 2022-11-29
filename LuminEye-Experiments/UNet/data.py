import os
import numpy as np
import cv2
import torch
from glob import glob
from torch.utils.data import Dataset
import albumentations as A 




class UBRIS(Dataset):
    def __init__(self,images_path,masks_path,transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_sample = len(images_path)
        self.transform = transform
        
    
    def __getitem__(self,idx):
        image = cv2.imread(self.images_path[idx],cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_path[idx],cv2.IMREAD_GRAYSCALE)
        # print(f"Mask Size {mask.shape} ")
        image = image.astype("uint8")
        mask = mask.astype("uint8")
        mask = np.expand_dims(mask,axis=2)
        
        if self.transform is not None:
            transformed = self.transform(image=image,mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            
            # print(f"Augmentation: |image: {image.size()} | mask: {mask.size()}")
            
            image = image.permute(2,0,1)
            image = image.float()/255
            mask = mask.permute(2,0,1)
            maks = mask.float()/255
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2, 0, 1)
            image = image.float()/255
            mask = mask.permute(2, 0, 1)
            mask = mask.float()/255
        print(f"Image shape {image.shape}| Mask shape {mask.shape}")
        return image,mask
    
    def __len__(self):
        return self.n_sample
    
    
if __name__ == '__main__':
    train_img = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/train_img"
    train_mask = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/train_masks"
    
    train_x = sorted(glob(f"{train_img}/*"))
    train_y = sorted(glob(f"{train_mask}/*"))
    # print(train_x)
    train_transform = A.Compose(
        [
            A.Resize(512,512),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                               rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                       b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5)
        ]
    )
        

    ubris = UBRIS(train_x,train_y,transform=train_transform)[5]
    
    
    

