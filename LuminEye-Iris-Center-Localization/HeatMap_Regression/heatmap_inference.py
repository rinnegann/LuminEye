import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
from torch.utils.data import Dataset
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
import mediapipe
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import math
import albumentations as A
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision import models
import wandb
from unet_model import UNET
from heatmap_regression CenterDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'



IMAGE_DIR = "/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD_mp2gaze/Images"
val_df = pd.read_csv("/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD_mp2gaze/mix_val.csv")

BACTH_SIZE = 4
test_ds = CenterDataset(val_df)

testLoader = DataLoader(test_ds, batch_size=BACTH_SIZE,
	num_workers=os.cpu_count(), pin_memory=True,drop_last=True)



def maskToKeypoints(mask):

    kp = np.unravel_index(np.argmax(mask, axis=None), dims=(64, 64))
    return kp[1], kp[0]


def findCoordinates(mask):

    hm_sum = np.sum(mask)

    index_map = [j for i in range(64) for j in range(64)]
    index_map = np.reshape(index_map, newshape=(64, 64))

    x_score_map = mask * index_map / hm_sum
    y_score_map = mask * np.transpose(index_map) / hm_sum

    px = np.sum(np.sum(x_score_map, axis=None))
    py = np.sum(np.sum(y_score_map, axis=None))

    return px, py


def calcRMSError(kps_gt, kps_preds):

    N = kps_gt.shape[0] * (kps_gt.shape[-1] // 2)
    error = np.sqrt(np.sum((kps_gt-kps_preds)**2)/N)

    return error


def calcKeypoints(model, gen):
    kps_gt = []
    kps_preds = []

    image_array = []
    gt_mask_array = []
    pred_mask_array = []
    nbatches = len(gen)

    unflatter = nn.Unflatten(1, (64, 64))

    model.eval()

    for (x, y) in gen:

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(x)

        for i in range(x.shape[0]):

            imgs = x[i].detach().cpu().permute(1, 2, 0).numpy()

            mask_gt = y[i]

            mask_gt = unflatten(mask_gt).detach(
            ).cpu().permute(1, 2, 0).numpy()

            mask_pred = y_pred[i]

            mask_pred = unflatten(mask_pred).detach(
            ).cpu().permute(1, 2, 0).numpy()

            xgt, ygt = findCoordinates(mask_gt[:, :, 0])

            xpred, ypred = findCoordinates(mask_pred[:, :, 0])

            image_array.append(imgs)
            gt_mask_array.append(mask_gt)
            pred_mask_array.append(mask_pred)

            kps_gt.append([xgt, ygt])
            kps_preds.append([xpred, ypred])

    return image_array, gt_mask_array, pred_mask_array, kps_gt, kps_preds


def showMasks(imgs, gt_masks, pred_masks, gt_coord, pred_coord, nrows=8, ncols=4):

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    plt.figure(figsize=(8, 8))

    axs[0, 0].set_title("Original Image")
    axs[0, 1].set_title("GT Masks")
    axs[0, 2].set_title("Pred Masks")
    axs[0, 3].set_title("Pred & GT Coord")

    for i in range(len(imgs)):

        axs[i, 0].imshow(imgs[i])
        axs[i, 0].axis("off")

        axs[i, 1].imshow(gt_masks[i])
        axs[i, 1].axis("off")

        axs[i, 2].imshow(pred_masks[i])
        axs[i, 2].axis("off")

        axs[i, 3].imshow(imgs[i])
        axs[i, 3].scatter(x=int(pred_coord[i][0]), y=int(
            pred_coord[i][1]), color='blue')  # Prediction Blue
        axs[i, 3].scatter(x=int(gt_coord[i][0]), y=int(
            gt_coord[i][1]), color="red")  # Ground Truth Red
        axs[i, 3].axis("off")

    plt.show()
    plt.close()


if __name__ == '__main__':

    model_path = ""

    model = torch.load(model_path)
    image_array, gt_mask_array, pred_mask_array, kps_gt, kps_preds = calcKeypoints(
        model, testLoader)

    low_limit = 50
    upper_limit = 58

    showMasks(image_array[low_limit:upper_limit], gt_mask_array[low_limit:upper_limit],
              pred_mask_array[low_limit:upper_limit], kps_gt[low_limit:upper_limit], kps_preds[low_limit:upper_limit])
