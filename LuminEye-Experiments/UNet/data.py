import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset




class UBRIS(Dataset):
    def __init__(self,images_path,masks_path,transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_sample = len(images_path)
        self.transform = transform
        
    def __getitem__(self,index):
        
        # Images
        image = cv2.imread(self.images_path[index],cv2.IMREAD_COLOR) # Read Image withough Transparent channels
        
        
        
        
        # Masks
        mask = cv2.imread(self.masks_path[index],cv2.IMREAD_GRAYSCALE)
        
        

        

        if self.transform is not None:
            transformed = self.transform(image=image,mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = cv2.resize(image,(512,512)) # Resize Image
            image = image/255.0 # (512,512,3)
            image = np.transpose(image,[2,0,1]) # (3,512,512)
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            
            
            
            mask = cv2.resize(mask,(512,512)) # Resize
            mask = mask/255 # (512,512)
            mask = np.expand_dims(mask,axis=0)  # (1,512,512)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
        return image, mask
    
    def __len__(self):
        return self.n_sample

