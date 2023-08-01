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
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose,initialize_config_dir
from utils import decode_segmap,UnNormalize
from datasets import Iris
sys.path.insert(1, "../utils/")
from losses import IoU, pixel_wise_accuracy, get_lr

model = torch.load("/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/DeepLabV3Plus/DeepLabv3_efficiennet_backbone_multiclass_boundary_awaweloss_epoch_200_batch_4/model-0.902.pt")

val_images = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Padded/val_image"

val_masks =  "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Padded/multi_classes" 



def config_initialize() -> None:
    
    with initialize_config_dir(version_base="1.3", config_dir="/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/DeepLabV3Plus"):

        cfg = compose(
            config_name="deeplab_config.yaml", return_hydra_config=True, overrides=[]
        )

    return cfg


config = config_initialize()
n_classes = config.MultiClassSegmentation.num_classes
batch_size = config.MultiClassSegmentation.batch_size
colors = config.MultiClassSegmentation.colors
label_colours = config.MultiClassSegmentation.label_colours
valid_classes = config.MultiClassSegmentation.valid_classes
class_names = config.MultiClassSegmentation.class_names
class_map = config.MultiClassSegmentation.class_map


valid_x = sorted(
        glob(f"{val_images}/*"))
valid_y = sorted(
    
        glob(f"{val_masks }/*"))

    
def predict_image_mask(model,image,mask):
    model.eval()
    
    image = image.to(device)
    mask = mask.to(device)
    
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



    
def get_images(test_x,test_y,val_transform,batch_size=1,shuffle=True,pin_memory=True):
    
    val_data  = Iris(test_x,test_y,transform =val_transform)
    test_batch = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,drop_last=True)
    return test_batch

val_transform = A.Compose([
    A.Resize(512,512),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_batch = get_images(valid_x,valid_y,val_transform,batch_size=batch_size)

val_cls  = Iris(valid_x,valid_y,transform =val_transform)


def main(saved_location,seperate=False):
    if not os.path.exists(saved_location):
        os.makedirs(saved_location)
    unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    total_iou = 0 
    for i in range(len(val_batch)):
        image,mask = val_cls[i]
        
        pred_mask,iou_score = predict_image_mask(model,image,mask)
        
        pred_mask = decode_segmap(pred_mask,n_classes,label_colours) * 255.0
        
        
        
        gt_mask = decode_segmap(mask,n_classes,label_colours) * 255.0
        
        img = unorm(image).permute(1,2,0).numpy() * 255.0
        
        if seperate:
            if not os.path.exists(os.path.join(saved_location,"img")):
                os.makedirs(os.path.join(saved_location,"img"))
                
            if not os.path.exists(os.path.join(saved_location,"pred")):
                os.makedirs(os.path.join(saved_location,"pred"))
                
            if not os.path.exists(os.path.join(saved_location,"mask")):
                os.makedirs(os.path.join(saved_location,"mask"))
                
            cv2.imwrite(os.path.join(saved_location,"img",f"{i}.png"),img[:,:,::-1])
            cv2.imwrite(os.path.join(saved_location,"pred",f"{i}.png"),pred_mask)
            cv2.imwrite(os.path.join(saved_location,"mask",f"{i}.png"),gt_mask)
        else:
            
            line = np.ones((512, 10, 3)) * 128
            
            
            cv2.putText(gt_mask,"GT",(10,10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            
            
            
            cat_images = np.concatenate(
                [img[:,:,::-1], line,pred_mask, line,gt_mask], axis=1
            )
            
            cv2.imwrite(os.path.join(saved_location,f"{i}.png"),cat_images)
        
        total_iou += iou_score
        
    # return total_iou
if __name__ == "__main__":
    experiment_name = "MyTestingExperiment"
    main(experiment_name,seperate=True)
    # print(f"Iou Value is {iou/len(val_batch)}")
