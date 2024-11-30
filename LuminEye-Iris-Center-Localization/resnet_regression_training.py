import os
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.optim import Adam
import pandas as pd
from torch.utils.data import DataLoader
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from BaseModels.resnetModels import BB_model
from utils import *
from dataset_classes import CenterDatasetRG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "/home/nipun/Desktop/Datasets/MixDataset/images/"
trn_df = pd.read_csv("/home/nipun/Desktop/Datasets/MixDataset/trainAll.csv")


# mask = np.random.randn(len(trn_df)) < 0.8


# val_df = trn_df[~mask]

val_df = pd.read_csv("/home/nipun/Desktop/Datasets/MixDataset/valAll.csv")


print(trn_df.head())
RESIZE_AMT = 64
BACTH_SIZE = 32

train_transforms = A.Compose(
    [
        A.Resize(width=RESIZE_AMT, height=RESIZE_AMT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(width=RESIZE_AMT, height=RESIZE_AMT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1),
    ]
)


train_ds = CenterDatasetRG(trn_df, transforms=train_transforms)
test_ds = CenterDatasetRG(val_df, transforms=val_transforms)

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


def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


def main_training(model, optimizer, scheduler, train_dl, test_dl, epochs, loss_fn):
    idx = 0

    prev_loss = 0

    early_stopping = EarlyStopping(verbose=True)
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0

        for x, y_bb in train_dl:
            batch = x.shape[0]
            x = x.cuda().float()

            y_bb = y_bb.cuda()

            out_bb = model(x)

            loss_bb = loss_fn(out_bb, y_bb).sum(1)
            # loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()

            loss = loss_bb

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx += 1

            total += batch

            sum_loss += loss.item()

        val_loss = val_epochs(model, test_dl, loss_fn)

        train_loss = sum_loss / total
        scheduler.step(train_loss)

        early_stopping(val_loss, model)

        train_loss = sum_loss / total
        scheduler.step(val_loss)

        my_lr = scheduler.get_lr()

        print(my_lr)

        print(f"Epoch Number {i+1}")
        print("train_loss %.3f " % (train_loss))
        print("Validation Loss %.3f " % (val_loss))
        print("*" * 8)

        if early_stopping.early_stop:
            print("Early stopping")

            train_metrics = {"train/epoch": i + 1, "train/train_loss": train_loss}

            val_metrics = {"val/epoch": i + 1, "val/val_loss": val_loss}
            wandb.log({**train_metrics, **val_metrics})
            break


def val_epochs(model, val_loader, loss_fn):

    model.eval()
    total_val_loss = 0
    total = 0
    for x, y_bb in val_loader:

        x = x.cuda().float()
        y_bb = y_bb.cuda()

        out_bb = model(x)

        total += x.shape[0]
        with torch.no_grad():
            # loss_bb = F.l1_loss(out_bb,y_bb,reduction='none').sum(1)
            loss_bb = loss_fn(out_bb, y_bb).sum(1)
            loss_bb = loss_bb.sum()

            total_val_loss += loss_bb.item()

    return total_val_loss / total


if __name__ == "__main__":

    n_epoch = 100

    config = {"epochs": n_epoch, "max_learning_rate": 0.006}

    experiment_name = f"Regression_EfficientNetWithCoordConv__epoch_{n_epoch}_smoothl1Loss_summation_batch_{BACTH_SIZE}_resize_{RESIZE_AMT}_for_gi4e_bioid_h2head_withKaimingInitialization"

    # experiment_name = "checkValidationDataInCorrectFormat"

    wandb.init(
        project="LuminEys-Iris", entity="rinnegann", name=experiment_name, config=config
    )

    # loss_fn = nn.MSELoss(reduction='none')
    # loss_fn = nn.L1Loss(reduction='none')

    loss_fn = nn.SmoothL1Loss(reduction="none")

    # loss_fn = WingLoss()

    # model = CoordEfficientModel(device=device).cuda()
    model = BB_model().cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.006)

    update_optimizer(optimizer, 0.001)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        'min',patience=1,cooldown=1)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config["max_learning_rate"],
        n_epoch,
        steps_per_epoch=len(train_ds) // BACTH_SIZE,
    )

    main_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dl=trainLoader,
        test_dl=testLoader,
        epochs=n_epoch,
        loss_fn=loss_fn,
    )
