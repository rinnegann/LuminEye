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


train_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/train_img"
train_masks  = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/train_masks"


val_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/val_img"

val_masks =  "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/val_masks" 




train_x = sorted(
        glob(f"{train_images}/*"))
train_y = sorted(
        glob(f"{train_masks}/*"))
valid_x = sorted(
        glob(f"{val_images}/*"))
valid_y = sorted(
        glob(f"{val_masks }/*"))


yolo_model_path = "/home/nipun/Music/yolov5/runs/train/exp8/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path ,force_reload=True)

# Filter BBOX Based on Confidence
model.conf = 0.40


saved_img_location = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/YoloCroppedEyes/images"
saved_mask_location = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/YoloCroppedEyes/masks"



if not os.path.exists(saved_img_location ):
    os.makedirs(saved_img_location)
    
    


if not os.path.exists(saved_mask_location):
    os.makedirs(saved_mask_location)
    
    
    
def main(images,masks,visualize=False):
    padded_amt = 5
    pad_top_left_y1 = 15
    pad_bottom_right_x2 = 7
    RESIZE_AMT = 256


    rows = len(valid_x)
    
    
    
    
    # fig,axes = plt.subplots(5,4,figsize=(15,15))



    # axes[0,0].set_title("YOLO Pred")
    # axes[0,1].set_title("GT Mask")
    # axes[0,2].set_title("Cropped Img")
    # axes[0,3].set_title("Cropped Mask")



    for z,(img_path,mask_path) in enumerate(zip(images,masks)):
        
        
        print(img_path)
        image = cv2.imread(img_path)
        
        h,w   = image.shape[:2]
        mask = cv2.resize(cv2.imread(mask_path),(h,w))
        
        
        
        results = model(image[:,:,::-1],size=640)
        
        df = results.pandas().xyxy[0]
                                    
                                    
        df = df[df["class"]==0] # Get the all BBOX related to Iris Class which is zero
        
        df = df[df['confidence'] == df['confidence'].max()]
        
        
        image_resized = cv2.resize(image,(RESIZE_AMT,RESIZE_AMT))
        mask_resized = cv2.resize(mask,(RESIZE_AMT,RESIZE_AMT))
        
        
        for (i,row) in df.iterrows():
            
            x1 = round((round(row["xmin"])/w) * RESIZE_AMT)
            y1 = round((round(row["ymin"])/h) * RESIZE_AMT)
            x2 = round((round(row["xmax"])/w) * RESIZE_AMT)
            y2 = round((round(row["ymax"])/h) * RESIZE_AMT)
        
    
       
        draw_image = image_resized.copy()
        
        if y1<pad_top_left_y1:
            pad_top_left_y1 = y1 
            
        elif x1<padded_amt:
            padded_amt = x1
        cropped_img = image_resized[y1-pad_top_left_y1:y2+padded_amt,x1-padded_amt:x2+pad_bottom_right_x2]
        
        cropped_mask = mask_resized[y1-pad_top_left_y1:y2+padded_amt,x1-padded_amt:x2+pad_bottom_right_x2]
        cropped_mask = np.where(cropped_mask>0,255,0)
        
        image_name = img_path.split("/")[-1]
        
        
        
        # if visualize:
            
            
        #     axes[z,0].imshow(draw_image[:,:,::-1])
        #     axes[z,1].imshow(mask_resized)
        #     axes[z,2].imshow(cropped_img[:,:,::-1])
        #     axes[z,3].imshow(cropped_mask)
    
    
        #     axes[z,0].axis("off")
        #     axes[z,1].axis("off")
        #     axes[z,2].axis("off")
        #     axes[z,3].axis("off")
            


        print(cropped_img.shape)
        print(cropped_mask.shape)
        cv2.imwrite(os.path.join(saved_img_location,image_name),cropped_img)
        cv2.imwrite(os.path.join(saved_mask_location,image_name),cropped_mask )
        
        
if __name__ == "__main__":
    main(train_x,train_y)
