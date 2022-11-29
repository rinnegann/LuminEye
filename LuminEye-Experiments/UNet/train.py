import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import UBRIS
from model import UNET
from loss import *
from utils import *
import math
import wandb
import sys
from rich.progress import track
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def train(model, loader, optimizer,epoch,device):
    epoch_loss = 0.0
    iou_score = 0
    accuracy = 0

    metric_monitor = MetricMonitor()
    model.train()

    stream = tqdm(loader)
    for i,(images,masks) in enumerate(stream,start=1):
        

        images =  images.to(device,dtype=torch.float32)
        masks = masks.to(device,dtype=torch.float32)
        
        # print(f"Train X size {images.size()}")
        # print(f"Train Y Size {masks.size()}")
        
        y_pred = model(images)
        
        
        # print(f"Y pred shape {y_pred.size()}")
        
        loss = DiceBceLoss(y_pred,masks)
        iou_score += IoU(y_pred,masks)
        accuracy = pixel_wise_accuracy(y_pred,masks)
        
        
        metric_monitor.update("Loss",loss.item())
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
        
        
        epoch_loss += loss.item()
        
    epoch_loss = epoch_loss/len(loader)
    iou_score = iou_score/len(loader)
    accuracy = accuracy/len(loader)

    return epoch_loss,iou_score,accuracy


def evaluate(model, loader,epoch,device):
    """ Caculate the Validation loss per epoch

    Args:
        model (pytorch model): Custom model
        loader (DataLoader): Validation DataLoader
        loss_fn (pytorch loss): Loss function
        epoch (int): epoch number
        device (str): Device that is supposed to train

    Returns:
        _float32_: validation loss per epoch
    """
    epoch_loss = 0.0
    iou_score = 0.0
    test_accuracy = 0
    
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(loader)
    with torch.no_grad():
        for i,(x, y) in enumerate(stream,start=1):
            x = x.to(device,dtype = torch.float32)
            y = y.to(device,dtype = torch.float32)

            y_pred = model(x)
            # loss = loss_fn(y_pred, y)
            loss = jaccard_loss(y_pred, y)
            iou_score += IoU(y_pred,y)
            test_accuracy += jaccard_loss(y_pred,y)
            
            metric_monitor.update("Loss", loss.item())
            epoch_loss += loss.item()
            
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        epoch_loss = epoch_loss/len(loader)
        iou_loss = iou_loss/len(loader)
        accuracy = accuracy/len(loader)
    return epoch_loss,iou_score,accuracy

def getting_trainval_local(trarin_img,train_mask,valid_img,valid_masks):
    """Getting training and validation images & masks from local

    Args:
        trarin_img (Str):  train Image Directory
        train_mask (Str): train Image Directory
        vali_img (str): val Image Directory
        val_masks (Str): val mask Directory

    Returns:
        List: List of both train and val data
    """
    
    
    train_x = sorted(
        glob(f"{train_img}/*"))
    train_y = sorted(
        glob(f"{train_mask}/*"))

    valid_x = sorted(
        glob(f"{valid_img}/*"))
    valid_y = sorted(
        glob(f"{valid_masks}/*"))

    if len(train_x) != len(train_y) or len(valid_x) != len(valid_y):
        print(f"Train Images {len(train_x)} Masks {len(train_y)}")
        print(f"Valid Images {len(valid_x)} Masks {len(valid_y)}")
        print("Found an incorrect number of training and valid datasets")
        sys.exit()

    print(
        f"Train Dataset Size: {len(train_x)} | Valid Dataset Size: {len(valid_x)}")
    
    return [train_x, train_y, valid_x, valid_y]

    
def create_train_val_dataLoader(train_x,train_y,valid_x,valid_y,config,visualize=False):
    
        train_transform = A.Compose(
        [
            A.Resize(config["image_height"],config["image_width"]),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                               rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                       b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5)
        ]
    )
        
        val_transform = A.Compose(
        [A.Resize(config["image_height"],config["image_width"])]
    )
        train_dataset = UBRIS(train_x, train_y, transform=train_transform)
        valid_dataset = UBRIS(valid_x, valid_y, transform=val_transform)

        if visualize:
            visualize_augmentations(train_dataset,idx=55)

        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=config["batch_size"],
                                shuffle=True,
                                num_workers=4
        )

        valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                num_workers=4)
        
        return train_loader, valid_loader
    
def main(*args):
    
    
    experiment_name = "Miche_With_Augmentation"
    
    
    seeding(42)

    create_dir("Unet_1st_Experiment")
    
    wandb.init(project="LuminEys-Iris",entity="rinnegann",name=experiment_name,config=config)
    
    train_x,train_y,val_x,val_y=getting_trainval_local(train_img, train_mask, valid_img, valid_masks)
    
    train_loader,val_loader = create_train_val_dataLoader(train_x, train_y, val_x, val_y,config)
    
    checkpoint_path = os.path.join("Unet_1st_Experiment",  config["model_fileName"])

    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config["batch_size"])

    print(f"Number of Steps for Epoch : {n_steps_per_epoch}")
    
    
    device = torch.device("cuda")
    model = UNET()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min",patience=5,verbose=True)
    # loss_fn = DiceBCELoss()
    
    
    
    best_valid_loss = float("inf")

    metrics = {}

    for epoch in range(config["epochs"]):
        start_time = time.time()

        train_loss,train_iou,train_accuracy= train(model, train_loader, optimizer, epoch, device)
        
        valid_loss,val_iou,val_accuracy = evaluate(model,val_loader,epoch ,device)

        train_metrics = {"train/epoch":epoch+1,"train/train_loss": train_loss,"train/accuracy": train_accuracy,"train/IoU":train_iou}

        
        val_metrics = {"val/epoch":epoch+1,"val/val_loss": valid_loss,"val/val_accuracy": val_accuracy,"val/IoU":val_iou}

        if valid_loss < best_valid_loss:
            print(f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}")

            best_valid_loss = valid_loss

            torch.save(model.state_dict(), checkpoint_path)

        wandb.log({**train_metrics, **val_metrics})
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    wandb.finish()
        



if __name__ == '__main__':
    
    config = {"image_height": 512, "image_width":512, "learning_rate": 1e-4, "epochs": 50,
              "batch_size": 2, "model_fileName": "miche_checkpoint_with_augmentations.pth", "Augmentations": {"train": ["Resize", "ShiftScaleRotate", "RGBSHIFT", "RandomBrightnessContrast", "Normalize", "To Tensor"], "val": ["Resize","Normalize","To Tensor"],}}


    train_img = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/train_img"
    train_mask = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/train_masks"
    
    valid_img = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/val_img"
    valid_masks = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/val_masks"
    
    
    main(train_img,train_mask,valid_img,valid_masks,config)

    
    

    

    

    
