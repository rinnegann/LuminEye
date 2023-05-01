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




model = torch.load("/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/Miche_model_2023_04_24_04:04:09_val_iou0.732.pt")

val_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/val_img"

val_masks =  "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/val_masks/" 
n_classes = 3
batch_size = 1

img_resize = 256

colors = [ [  0,   0,   0],[0,255,0],[0,0,255]]
label_colours = dict(zip(range(n_classes), colors))

valid_classes = [0,85, 170]
class_names = ["Background","Pupil","Iris"]


class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)


def decode_segmap(temp):
    #convert gray scale to color
    temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb




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
    
    
def predict_image_mask(model,image,mask):
    model.eval()
    
    image = image.to(device)
    mask = mask.to(device)
    
    # print(f"Original Image shape: {image.size()}")
    
    # print(f"Ground Truth Mask shape: {mask.size()}")
    
    with torch.no_grad():
        
        softmax = nn.Softmax(dim=1)
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        model_output,_,_,_,_,_,_ = model(image)
        
        
        output = softmax(model_output)
        score = IoU(model_output, mask)
        
        masked = torch.argmax(output,dim=1)
        masked = masked.cpu().squeeze(0)
    return masked,score


class Iris(Dataset):
    def __init__(self,images,masks,transform = None):
        self.transforms = transform
        self.images = images
        self.masks = masks
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        
        image = Image.open(self.images[index])
        img = np.array(image.resize((64,64)))
        
        mask = Image.open(self.masks[index])
        mask = np.array(mask.resize((256,256)))
        
               
        if self.transforms is not None:
            aug = self.transforms(image=img,mask=mask)
            img = aug['image']
            mask = aug['mask']
        return img,mask
    
    

    
def get_images(test_x,test_y,val_transform,batch_size=1,shuffle=True,pin_memory=True):
    
    val_data  = Iris(test_x,test_y,transform =val_transform)
    test_batch = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,drop_last=True)
    return test_batch

transform = A.Compose([
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])



val_batch = get_images(valid_x,valid_y,transform,batch_size=batch_size)

val_cls  = Iris(valid_x,valid_y,transform =transform)


def main(saved_location):
    if not os.path.exists(saved_location):
        os.makedirs(saved_location)
    unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    total_iou = 0 
    for i in range(len(val_batch)):
        image,mask = val_cls[i]
        
        pred_mask,iou_score = predict_image_mask(model,image,mask)
        
        pred_mask = decode_segmap(pred_mask) * 255.0
        
        print(pred_mask.shape)
        
        gt_mask = decode_segmap(mask) * 255.0
        
        print(gt_mask.shape)
        
        img = unorm(image).permute(1,2,0).numpy() * 255.0
        
        
        img = cv2.resize(img,(256,256))
        
        line = np.ones((img_resize, 10, 3)) * 128
        
        
        cv2.putText(gt_mask,"GT",(10,10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        
        
        
        cat_images = np.concatenate(
            [img[:,:,::-1], line,pred_mask, line,gt_mask], axis=1
        )
        
        cv2.imwrite(os.path.join(saved_location,f"{i}.png"),cat_images)
        
        total_iou += iou_score
        
    return total_iou


if __name__ == "__main__":
    experiment_name = "Predictions/u2net_light_with_ersgan"
    iou = main(experiment_name)
    print(f"Iou Value is {iou/len(val_batch)}")
