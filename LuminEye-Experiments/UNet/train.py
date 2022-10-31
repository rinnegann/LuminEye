import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import UBRIS
from model import UNET
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time


def train(model,loader,optimizer,loss_fn,device):
    epoch_loss  = 0.0
    
    model.train()
    for x,y in loader:
        x = x.to(device,dtype=torch.float32)
        y = y.to(device,dtype=torch.float32)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == '__main__':

    seeding(42)

    create_dir("Unet_1st_Experiment")

    # Load train,valid datasets
    train_x = sorted(
        glob("/home/nipun/Documents/Uni_Malta/Datasets/Datasets/UBRIS/train_img/*"))
    train_y = sorted(
        glob("/home/nipun/Documents/Uni_Malta/Datasets/Datasets/UBRIS/train_masks/*"))

    valid_x = sorted(
        glob("/home/nipun/Documents/Uni_Malta/Datasets/Datasets/UBRIS/val_img/*"))
    valid_y = sorted(
        glob("/home/nipun/Documents/Uni_Malta/Datasets/Datasets/UBRIS/val_masks/*"))

    print(
        f"Train Dataset Size: {len(train_x)} | Valid Dataset Size: {len(valid_x)}")

    
    # Hyperparameters
    H = 512
    W = 512
    
    size = (H,W)
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = os.path.join("Unet_1st_Experiment","checkpoint.pth")
    
    train_dataset = UBRIS(train_x,train_y)
    valid_dataset = UBRIS(valid_x,valid_y)
    
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2
    )
    
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=2
    )
    
    
    device = torch.device("cuda")
    model = UNET()
    model = model.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min",patience=5,verbose=True)
    loss_fn = DiceBCELoss()
    
    
    best_valid_loss = float("inf")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train(model,train_loader,optimizer,loss_fn,device)
        valid_loss = evaluate(model,valid_loader, loss_fn, device)
        
        if valid_loss < best_valid_loss:
            print(f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}")
            
            best_valid_loss = valid_loss
            
            torch.save(model.state_dict(), checkpoint_path)
            
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
        