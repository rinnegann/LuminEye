"""
This file is created to specifically to train UBRIS V2 dataset

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob

import albumentations as A
import wandb
import time
from tqdm.notebook import tqdm
from custom_model import U2NET
from boundary_loss import DC_and_BD_loss
import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.insert(1, "../")
from utils import *
from dataset_classes import UBRIS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Available devices:{device}")

n_classes = 2
RESIZE_AMT = 256

train_images = (
    "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/YoloCroppedEyes/train_imgs"
)
train_masks = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/YoloCroppedEyes/train_masks/"


val_images = (
    "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/YoloCroppedEyes/val_imgs/"
)

val_masks = (
    "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/YoloCroppedEyes/val_masks/"
)
train_x = sorted(glob(f"{train_images}/*"))
train_y = sorted(glob(f"{train_masks}/*"))
valid_x = sorted(glob(f"{val_images}/*"))
valid_y = sorted(glob(f"{val_masks }/*"))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


train_transform = A.Compose(
    [
        A.Resize(RESIZE_AMT, RESIZE_AMT),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
    ]
)

val_transform = A.Compose([A.Resize(RESIZE_AMT, RESIZE_AMT)])


train_set = UBRIS(train_x, train_y, mean, std, transform=train_transform)
val_set = UBRIS(valid_x, valid_y, mean, std, transform=val_transform)

batch_size = 2
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

x, y = next(iter(train_loader))

print(f" x = shape : {x.shape} ; type :{x.dtype}")
print(f" x = min : {x.min()} ; max : {x.max()}")
print(f" y = shape: {y.shape}; class : {y.unique()}; type: {y.dtype}")


def fit(epochs, model, train_loader, val_loader, optimizer, scheduler, patch=False):
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        running_target_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward

            y0, y1, y2, y3, y4, y5, y6 = model(image)
            # output = model(image)

            # Multi DiceBceLoss
            loss_1, loss = multi_dice_loss_function(y0, y1, y2, y3, y4, y5, y6, mask)

            # Multi Focal losss
            # loss_1,loss = multi_focal_losss(y0, y1, y2, y3, y4, y5, y6, mask)

            # #Multi Tversky loss
            # loss_1,loss = multi_tversky_loss(y0, y1, y2, y3, y4, y5, y6, mask)

            # Multi Distance Binary Loss
            # loss_1,loss = multi_distBinary_loss(y0, y1, y2, y3, y4, y5, y6, mask)

            # Multi Focal Tversky Loss
            # loss_1,loss  = multi_focaltversky_loss(y0, y1, y2, y3, y4, y5, y6, mask)

            # evaluation metrics
            iou_score += IoU(y0, mask)
            accuracy += pixel_wise_accuracy(y0, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()
            running_target_loss += loss_1.item()

        else:
            model.eval()
            test_loss = 0
            test_target_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    y0_l, y1_l, y2_l, y3_l, y4_l, y5_l, y6_l = model(image)
                    # evaluation metrics
                    val_iou_score += IoU(y0_l, mask)
                    test_accuracy += pixel_wise_accuracy(y0_l, mask)
                    # loss

                    loss_1, loss = multi_dice_loss_function(
                        y0_l, y1_l, y2_l, y3_l, y4_l, y5_l, y6_l, mask
                    )

                    # loss_1,loss = multi_distBinary_loss(y0_l, y1_l, y2_l, y3_l, y4_l, y5_l, y6_l, mask)

                    # loss_1,loss = multi_focaltversky_loss(y0_l, y1_l, y2_l, y3_l, y4_l, y5_l, y6_l, mask)

                    test_loss += loss.item()
                    test_target_loss += loss_1.item()

            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print(
                    "Loss Decreasing.. {:.3f} >> {:.3f} ".format(
                        min_loss, (test_loss / len(val_loader))
                    )
                )
                min_loss = test_loss / len(val_loader)
                decrease += 1
                if decrease % 5 == 0:
                    print("saving model...")
                    model_name = "{}_model_{}val_iou{:.3f}.pt".format(
                        "Miche",
                        get_current_date_time(),
                        val_iou_score / len(val_loader),
                    )

                    torch.save(model, model_name)

            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print(
                "Epoch:{}/{}..".format(e + 1, epochs),
                "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                "Train Target Loss: {:.3f}..".format(
                    running_target_loss / len(train_loader)
                ),
                "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                "Val Target Loss: {:.3f}..".format(test_target_loss / len(val_loader)),
                "Train IoU:{:.3f}..".format(iou_score / len(train_loader)),
                "Val IoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                "Time: {:.2f}m".format((time.time() - since) / 60),
            )

            train_metrics = {
                "train/epoch": e + 1,
                "train/train_loss": running_loss / len(train_loader),
                "train/train_target_loss": running_target_loss / len(train_loader),
                "train/iou": iou_score / len(train_loader),
                "train/accuracy": accuracy / len(train_loader),
            }

            val_metrics = {
                "val/epoch": e + 1,
                "val/var_target_loss": test_target_loss / len(val_loader),
                "val/val_loss": test_loss / len(val_loader),
                "val/iou": val_iou_score / len(val_loader),
                "val/accuracy": test_accuracy / len(val_loader),
            }

            wandb.log({**train_metrics, **val_metrics})

    wandb.log({**train_metrics, **val_metrics})
    wandb.finish()
    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))


if __name__ == "__main__":

    max_lr = 1e-3
    epoch = 50
    weight_decay = 1e-6

    experiment_name = "Short_Experiment"

    model = U2NET(in_ch=3, out_ch=2)

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=max_lr, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_loader)
    )

    config = {
        "epochs": epoch,
        "max_learning_rate": max_lr,
        "length_train": len(train_set),
        "length_val": len(val_set),
        "Augmentations": [
            "ShiftScaleRotate",
            "RGBShift",
            "RandomBrightnessContrast",
            "Blur",
            "MotionBLur",
            "MedianBlur",
        ],
        "number_of_classes": n_classes,
        "Resize_amt": (512, 512),
        "Base Model": "DeepLabV3Plus",
        "BackBone": "EfficientNet-B3",
        "Dataset": "Short",
        "OPtimizer": "Adam",
        "lr_scheduler": "OneCycleLR",
        "Loss": "Tversky Loss",
        "weight_decay": weight_decay,
    }

    wandb.init(
        project="LuminEys-Iris", entity="rinnegann", name=experiment_name, config=config
    )

    history = fit(epoch, model, train_loader, val_loader, optimizer, sched)
