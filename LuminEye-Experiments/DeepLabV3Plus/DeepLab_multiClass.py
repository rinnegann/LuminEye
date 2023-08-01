
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
from torch.optim import Adam
from glob import glob
import segmentation_models_pytorch as smp
import time
import wandb
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose,initialize_config_dir
from utils import UnNormalize,fit
from datasets import Iris
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = 'cpu'


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


train_images = config.MultiClassSegmentation.train_img_path
train_masks = config.MultiClassSegmentation.train_mask_path


val_images = config.MultiClassSegmentation.val_img_path
val_masks = config.MultiClassSegmentation.val_mask_path


train_x = sorted(
    glob(f"{train_images}/*"))
train_y = sorted(
    glob(f"{train_masks}/*"))
valid_x = sorted(
    glob(f"{val_images}/*"))
valid_y = sorted(
    glob(f"{val_masks }/*"))





def get_images(train_x, train_y, test_x, test_y, train_transform, val_transform, batch_size=1, shuffle=True, pin_memory=True):
    train_data = Iris(train_x, train_y, transform=train_transform)
    val_data = Iris(test_x, test_y, transform=val_transform)
    train_batch = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_batch = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_batch, test_batch


train_transform = A.Compose([
    A.Resize(config.MultiClassSegmentation.resize_amt,config.MultiClassSegmentation.resize_amt),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                               rotate_limit=30, p=0.5),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                       b_shift_limit=25, p=0.5),
    A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.MotionBlur(p=0.3),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()])

val_transform = A.Compose([
    A.Resize(512,512),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


train_batch, val_batch = get_images(
    train_x, train_y, valid_x, valid_y, train_transform, val_transform, batch_size=batch_size)


def visualize_multiclass_mask(train_batch):
    for img, mask in train_batch:
        print(img.shape)
        img1 = np.transpose(img[0, :, :, :], (1, 2, 0))
        mask1 = np.array(mask[0, :, :])
        img2 = np.transpose(img[1, :, :, :], (1, 2, 0))
        mask2 = np.array(mask[1, :, :])
        img3 = np.transpose(img[2, :, :, :], (1, 2, 0))
        mask3 = np.array(mask[2, :, :])
        fig, ax = plt.subplots(3, 2, figsize=(18, 18))
        ax[0][0].imshow(img1)
        ax[0][1].imshow(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].imshow(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].imshow(mask3)
        break


x, y = next(iter(train_batch))

print(f' x = shape : {x.shape} ; type :{x.dtype}')
print(f' x = min : {x.min()} ; max : {x.max()}')
print(f' y = shape: {y.shape}; class : {y.unique()}; type: {y.dtype}')




if __name__ == "__main__":
    max_lr = config.MultiClassSegmentation.max_lr
    epoch = config.MultiClassSegmentation.num_epochs
    weight_decay = config.MultiClassSegmentation.weight_decay

    model = smp.DeepLabV3Plus(config.MultiClassSegmentation.model_name, encoder_weights='imagenet', classes=n_classes, encoder_output_stride=16, activation=None,
                              encoder_depth=5)
    model = model.to(device)
    experiment_name = config.MultiClassSegmentation.experiment_name

    optimizer = torch.optim.Adam(
        model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_batch))

    config = {"epochs": config.MultiClassSegmentation.num_epochs,
              "max_learning_rate": config.MultiClassSegmentation.max_lr,
              "length_train": len(train_batch),
              "length_val": len(val_batch),
              "Augmentations": ["ShiftScaleRotate", "RGBShift", "RandomBrightnessContrast", "MotionBLur"],
              "number_of_classes": config.MultiClassSegmentation.num_classes,
              "Resize_amt": (config.MultiClassSegmentation.resize_amt, config.MultiClassSegmentation.resize_amt),
              "Base Model": "DeepLabV3Plus",
              "BackBone": config.MultiClassSegmentation.model_name,
              "Dataset": "Short",
              "OPtimizer": "Adam",
              "lr_scheduler": "OneCycleLR",
              "Loss": "DiceLoss",
              "weight_decay": weight_decay}

    wandb.init(project="LuminEys-Iris", entity="rinnegann",
               name=experiment_name,
               config=config)

    history = fit(epoch, model, train_batch, val_batch, optimizer, sched,class_names)
