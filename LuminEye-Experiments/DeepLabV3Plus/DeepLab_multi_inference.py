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
import time 
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")




model = torch.load("/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/UNet/model-0.845.pt")

train_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/train_img/"
train_masks  = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/train_masks/"


val_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/val_img"

val_masks =  "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/val_masks/" 
n_classes = 3
batch_size = 2

train_x = sorted(
        glob(f"{train_images}/*"))
train_y = sorted(
        glob(f"{train_masks}/*"))
valid_x = sorted(
        glob(f"{val_images}/*"))
valid_y = sorted(
        glob(f"{val_masks }/*"))


class Iris(Dataset):
    def __init__(self,images,masks,transform = None):
        self.transforms = transform
        self.images = images
        self.masks = masks
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))
        if self.transforms is not None:
            aug = self.transforms(image=img,mask=mask)
            img = aug['image']
            mask = aug['mask']
        return img,mask
    
def get_images(train_x,train_y,test_x,test_y,train_transform,val_transform,batch_size=1,shuffle=True,pin_memory=True):
    train_data = Iris(train_x,train_y,transform = train_transform)
    val_data  = Iris(test_x,test_y,transform =val_transform)
    train_batch = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_batch = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,drop_last=True)
    return train_batch,test_batch




train_transform = A.Compose([
    A.Resize(512,512),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                               rotate_limit=30, p=0.5),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                       b_shift_limit=25, p=0.5),
    A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.MotionBlur(p=0.3),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(512,512),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

train_batch,val_batch = get_images(train_x,train_y,valid_x,valid_y,train_transform,val_transform,batch_size=batch_size)

for x,y in val_batch:
    x = x.to(device)
    fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
    softmax = nn.Softmax(dim=1)
    preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
    img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
    preds1 = np.array(preds[0,:,:])
    mask1 = np.array(y[0,:,:])
    img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
    preds2 = np.array(preds[1,:,:])
    mask2 = np.array(y[1,:,:])
    # img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
    # preds3 = np.array(preds[2,:,:])
    # mask3 = np.array(y[2,:,:])
    ax[0,0].set_title('Image')
    ax[0,1].set_title('Prediction')
    ax[0,2].set_title('Mask')
    ax[1,0].set_title('Image')
    ax[1,1].set_title('Prediction')
    ax[1,2].set_title('Mask')
    ax[2,0].set_title('Image')
    ax[2,1].set_title('Prediction')
    ax[2,2].set_title('Mask')
    ax[0][0].axis("off")
    ax[1][0].axis("off")
    ax[2][0].axis("off")
    ax[0][1].axis("off")
    ax[1][1].axis("off")
    ax[2][1].axis("off")
    ax[0][2].axis("off")
    ax[1][2].axis("off")
    ax[2][2].axis("off")
    ax[0][0].imshow(img1)
    ax[0][1].imshow(preds1)
    ax[0][2].imshow(mask1)
    ax[1][0].imshow(img2)
    ax[1][1].imshow(preds2)
    ax[1][2].imshow(mask2)
    # ax[2][0].imshow(img3)
    # ax[2][1].imshow(preds3)
    # ax[2][2].imshow(mask3)
    plt.show()   
    break