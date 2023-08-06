from datetime import datetime
import numpy as np
import os
import pandas as pd
import time
import wandb

import torch
import torch.nn as nn
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import sys
sys.path.insert(1, "../utils/")
from losses import DiceLoss, IoU, pixel_wise_accuracy, get_lr


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


def get_current_date_time():
    now = datetime.now()
    year = now.strftime("%Y")

    month = now.strftime("%m")

    day = now.strftime("%d")

    time_n = now.strftime("%H:%M:%S")

    return f"{year}_{month}_{day}_{time_n}_"


def decode_segmap(temp, n_classes, label_colours):

    # print(temp.size())
    temp = temp.numpy()

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


def labels(class_names):
    l = {}
    for i, label in enumerate(class_names):
        l[i] = label
    return l





def wandb_mask(bg_img, pred_mask, true_mask,class_names):
    return wandb.Image(bg_img, masks={"prediction": {
        "mask_data": pred_mask,
        "class_labels": labels(class_names)
    }, "ground truth": {
        "mask_data": true_mask,
        "class_labels": labels(class_names)
    }})




def prediction_on_val(model, images, predictions, masks,class_names):

    mask_list = []

    count = 0

    softmax = nn.Softmax(dim=1)
    unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    for x, y, z in zip(images, predictions, masks):

        for i in range(x.shape[0]):

            img = unorm(x[i]).permute(1, 2, 0).cpu().numpy()

            mask = z[i].cpu().numpy()
0
            pred = softmax(y[i])  # [3, 512, 512]

            pred = torch.argmax(pred, dim=0).cpu().numpy()

            mask_list.append(wandb_mask(img, pred, mask,class_names))

    return mask_list




def fit(epochs, model, train_loader, val_loader, optimizer, scheduler, class_names,patch=False):

    dice_loss = DiceLoss(mode="multiclass")
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
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
            
          
            output = model(image)
            
            
            
         
           
            loss = dice_loss(output, mask)
            
           
            # evaluation metrics
            iou_score += IoU(output, mask)
            accuracy += pixel_wise_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            wb_images = []
            wb_prediction = []
            wb_masks = []
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data
                    image_titles = image_tiles.to(device)
                    mask_tiles = mask_tiles .to(device)

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)

                    wb_images.append(image)
                    wb_masks.append(mask)
                    output = model(image)
                    wb_prediction.append(output)
                    # evaluation metrics
                    val_iou_score += IoU(output, mask)
                    test_accuracy += pixel_wise_accuracy(output, mask)
                    # loss
                    loss = dice_loss(output, mask)
                    test_loss += loss.item()

            # calculatio mean for each batch
            mask_list = prediction_on_val(
                model, wb_images, wb_prediction, wb_masks,class_names)
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))

            if min_loss > (test_loss/len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(
                    min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                print('saving model...')
                torch.save(
                    model, 'model-{:.3f}.pt'.format(val_iou_score/len(val_loader)))

            # iou
            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(
                      running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                  "Train IoU:{:.3f}..".format(iou_score/len(train_loader)),
                  "Val IoU: {:.3f}..".format(val_iou_score/len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))

            train_metrics = {"train/epoch": e+1, "train/train_loss": running_loss/len(
                train_loader), "train/iou": iou_score/len(train_loader), "train/accuracy": accuracy/len(train_loader)}

            val_metrics = {"train/epoch": e+1, "val/val_loss": test_loss/len(
                val_loader), "val/iou": val_iou_score/len(val_loader), "val/accuracy": test_accuracy/len(val_loader)}

            wandb.log({"predictions": mask_list})
            wandb.log({**train_metrics, **val_metrics})

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}

    wandb.finish()
    print('Total time: {:.2f} m' .format((time.time() - fit_time)/60))
    return history



if __name__ == "__main__":
    pass
