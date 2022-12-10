import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from glob import glob
from datetime import datetime

from PIL import Image
import cv2
import albumentations as A
import wandb
import time 
from tqdm.notebook import tqdm
from torchsummary import summary
import segmentation_models_pytorch as smp
from model import unet_swin
import warnings
warnings.filterwarnings("ignore")

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes = 2


def get_current_date_time():
    now = datetime.now()
    year = now.strftime("%Y")


    month = now.strftime("%m")


    day = now.strftime("%d")

    time = now.strftime("%H:%M:%S")
    
    return f"{year}_{month}_{day}_{time}_"

def dense_target(tar:np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    
    for idx,value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx
    
    return dummy

class UBRIS:
    def __init__(self,image_path,target_path,mean,std,transform=None,test=False):
        
        self.image_path = image_path
        self.target_path = target_path
        self.mean = mean
        self.std = std
        self.transform = transform
        self.test = test
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self,idx):
        img = cv2.resize(cv2.cvtColor(cv2.imread(self.image_path[idx]) , cv2.COLOR_BGR2RGB),(256,256))
        target = cv2.resize(cv2.imread(self.target_path[idx] , cv2.IMREAD_GRAYSCALE),(256,256))
        
        target = np.where( target > 0,255,0)
        
        
        if self.transform is not None:
            aug = self.transform(image= img,target = target)
            img = Image.fromarray(aug["image"])
            target = aug["target"]
            
        if self.transform is None:
            img = Image.fromarray(img)
            
            
        t = T.Compose([T.ToTensor() , T.Normalize(self.mean , self.std)])
        
        if self.test is False:
            img = t(img)
        
        target = dense_target(target)
        target = torch.from_numpy(target).long()
        
        return img,target
    
mean = [0.485 ,0.456 ,0.406]
std = [0.229 , 0.224 , 0.225]


train_images = "/home/nipun/Documents/Uni_Malta/Datasets/ShortDatasetTesting/train_img"
train_masks  = "/home/nipun/Documents/Uni_Malta/Datasets/ShortDatasetTesting/train_masks"


val_images = "/home/nipun/Documents/Uni_Malta/Datasets/ShortDatasetTesting/val_img"

val_masks =  "/home/nipun/Documents/Uni_Malta/Datasets/ShortDatasetTesting/val_masks" 


train_x = sorted(
        glob(f"{train_images}/*"))
train_y = sorted(
        glob(f"{train_masks}/*"))
valid_x = sorted(
        glob(f"{val_images}/*"))
valid_y = sorted(
        glob(f"{val_masks }/*"))

train_transform = A.Compose(
        [
            A.Resize(256,256),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                               rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                       b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5)
        ]
    )
        
val_transform = A.Compose(
        [A.Resize(256,256)]
    )


train_set = UBRIS(train_x,train_y,mean, std,transform=train_transform)
val_set = UBRIS(valid_x ,valid_y,mean , std,transform=val_transform)

batch_size = 2
train_loader= DataLoader(train_set , batch_size= batch_size , shuffle =True,drop_last=True)
val_loader = DataLoader(val_set , batch_size = batch_size , shuffle =False,drop_last=True)

x , y =next(iter(train_loader))

print(f' x = shape : {x.shape} ; type :{x.dtype}')
print(f' x = min : {x.min()} ; max : {x.max()}')
print(f' y = shape: {y.shape}; class : {y.unique()}; type: {y.dtype}')



model = unet_swin(img_size=256,size="swinv2_base_window16_256")

model  = model.to(device)


# print(summary(model,input_size=(3,512,512)))

def pixel_wise_accuracy(output , mask):
  with torch.no_grad():
    output = torch.argmax(F.softmax(output , dim =1) , dim=1)
    correct = torch.eq(output , mask).int()
    accuracy = float(correct.sum())/ float(correct.numel())#total number
  return accuracy

def IoU(pred , true_pred , smooth =1e-10 , n_classes=2):
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

def DiceBceLoss(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.to("cpu").squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = 1- ((2.*intersection + eps)/(cardinality + eps)).mean()
    bce = F.cross_entropy(logits, true , reduction ="mean")
    dice_bce = bce + dice_loss
    return dice_bce

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model, train_loader, val_loader, optimizer, scheduler, patch=False):
    train_losses = []
    test_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        #training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            #training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1,c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)
            
            image = image_tiles.to(device); mask = mask_tiles.to(device);
            #forward
            output = model(image)
            loss = DiceBceLoss(mask, output)
            #evaluation metrics
            iou_score += IoU(output, mask)
            accuracy += pixel_wise_accuracy(output, mask)
            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
            
            #step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step() 
            
            running_loss += loss.item()
            
        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            #validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    #reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1,c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)
                    
                    image = image_tiles.to(device); mask = mask_tiles.to(device);
                    output = model(image)
                    #evaluation metrics
                    val_iou_score +=  IoU(output, mask)
                    test_accuracy += pixel_wise_accuracy(output, mask)
                    #loss
                    loss = DiceBceLoss(mask, output)                                  
                    test_loss += loss.item()
            
            #calculatio mean for each batch
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))


            if min_loss > (test_loss/len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    model_name = "{}_model_{}val_iou{:.3f}.pt".format("Miche",get_current_date_time(),val_iou_score/len(val_loader))
                    
                    torch.save(model,model_name)
                    

           
            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy/ len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                  "Train IoU:{:.3f}..".format(iou_score/len(train_loader)),
                  "Val IoU: {:.3f}..".format(val_iou_score/len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))
            
            train_metrics = {"train/epoch":e+1,"train/train_loss":running_loss/len(train_loader),"train/iou":iou_score/len(train_loader),"train/accuracy":accuracy/len(train_loader)}
    
            val_metrics = {"train/epoch":e+1,"val/val_loss":test_loss/len(val_loader),"val/iou":val_iou_score/len(val_loader),"val/accuracy":test_accuracy/len(val_loader)}

            
            wandb.log({**train_metrics, **val_metrics})
    
    wandb.log({**train_metrics, **val_metrics})
    wandb.finish()
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    


max_lr = 1e-3
epoch = 10
weight_decay = 1e-6

experiment_name = "Short_Experiment"

optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))



config = {"epochs":epoch,
                   "max_learning_rate":max_lr,
                   "length_train":len(train_set),
                   "length_val": len(val_set),
                   "Augmentations":["ShiftScaleRotate","RGBShift","RandomBrightnessContrast"],
                   "number_of_classes":n_classes,
                   "Resize_amt":(512,512),
                   "Base Model": "DeepLabV3Plus",
                   "BackBone":"EfficientNet-B3",
                   "Dataset": "Short",
                   "OPtimizer":"Adam",
                   "lr_scheduler": "OneCycleLR",
                   "weight_decay":weight_decay}


wandb.init(project="LuminEys-Iris",entity="rinnegann",
           name=experiment_name,
           config=config)

history = fit(epoch, model, train_loader, val_loader, optimizer, sched)

