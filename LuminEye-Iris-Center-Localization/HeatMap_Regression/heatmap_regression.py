import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from torchvision import models
import wandb
import sys

sys.path.append("../")

from BaseModels.unet_model import UNET
from dataset_classes import CenterDatasetHM

device = "cuda" if torch.cuda.is_available() else "cpu"


BACTH_SIZE = 32

RESIZE_AMT = 64

IMAGE_DIR = "/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD_mp2gaze/Images"


trn_df = pd.read_csv(
    "/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD_mp2gaze/mix_train.csv"
)
val_df = pd.read_csv(
    "/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/Mix_Iris_Center_Gi42_BioId_H2HEAD_mp2gaze/mix_val.csv"
)


def mean_squared_error(y_true, y_pred):
    """Return MSE for the Batch"""
    return torch.sum(torch.square(y_pred - y_true), axis=-1).mean()


train_ds = CenterDatasetHM(trn_df)
test_ds = CenterDatasetHM(val_df)

trainLoader = DataLoader(
    train_ds,
    batch_size=BACTH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=True,
)
testLoader = DataLoader(
    test_ds,
    batch_size=BACTH_SIZE,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=True,
)


def trainStep(model, trainLoader, optimizer):

    model.train()

    epoch_loss = 0

    total_step = 0

    for _, (x, y) in enumerate(trainLoader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = mean_squared_error(y, y_pred)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        total_step += 1

    return epoch_loss / total_step


def valStep(model, testLoader):
    model.eval()

    total_val_mse_loss = 0
    total_val_jaccard_index = 0

    total_step = 0

    for x, y in testLoader:

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():

            y_pred = model(x)

        # MSE
        loss = mean_squared_error(y, y_pred).item()

        total_val_mse_loss += loss

        total_step += 1

    return total_val_mse_loss / total_step


def main(model, trainLoader, testLoader, optimizer, epochs=100):

    val_loss = 0

    for epoch in range(epochs):

        train_epoch_mse_loss = trainStep(model, trainLoader, optimizer)
        val_mse, val_jaccard = valStep(model, testLoader)

        if epoch == 0:
            val_loss = val_mse

        elif val_loss < val_mse and abs(val_loss - val_mse) > 0.2:

            model_name = f"hm_model_{str(val_loss)}.pth"
            torch.save(model, model_name)

        print(f"Epoch {epoch+1}| Train MSE Loss--> {train_epoch_mse_loss}")
        print(f"Epoch {epoch+1}| VAL MSE Loss--> {val_mse}")
        print(f"Epoch {epoch+1}| VAL Jaccard Loss--> {val_jaccard}\n")

        train_metrics = {
            "train/epoch": epoch + 1,
            "train/train_MSE_loss": train_epoch_mse_loss,
        }
        val_metrics = {
            "val/epoch": epoch + 1,
            "val/val_MSE_loss": val_mse,
            "val/val_Jaccard": val_jaccard,
        }
        wandb.log({**train_metrics, **val_metrics})


if __name__ == "__main__":

    n_epoch = 500
    n_classes = 1

    config = {"epochs": n_epoch, "max_learning_rate": 0.006}

    wandb.init(
        project="LuminEys-Iris",
        entity="rinnegann",
        name=f"HM_Regression__epoch_{n_epoch}_mse_mean_batch_{BACTH_SIZE}_resize_{RESIZE_AMT}_for_gi4e_bioid_h2head_mp2gaze",
        config=config,
    )

    model = UNET(n_classes).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(parameters, lr=0.006)

    main(model, trainLoader, testLoader, optimizer, epochs=n_epoch)
