import cv2
import time
import numpy as nn
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image


class IrisV1(Dataset):
    def __init__(self, images, masks, transform=None, enhance=True):
        self.transforms = transform
        self.images = images
        self.masks = masks
        self.enhance = enhance

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.enhance:
            img = cv2.imread(self.images[index])
            img, _ = upsampler.enhance(img, outscale=2)
            img = np.array(img[:, :, ::-1])

        else:
            img = np.array(Image.open(self.images[index]))

        h, w = img.shape[:2]

        mask = np.array(Image.open(self.masks[index]))
        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        return img, mask, (h, w), self.images[index].split("/")[-1]


class Iris(Dataset):
    def __init__(self, images, masks, transform=None):
        self.transforms = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))
        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        return img, mask


class UBRIS(Dataset):
    def __init__(self, image_path, target_path, mean, std, transform=None, test=False):
        super().__init__()

        self.image_path = image_path
        self.target_path = target_path
        self.mean = mean
        self.std = std
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img = cv2.resize(
            cv2.cvtColor(cv2.imread(self.image_path[idx]), cv2.COLOR_BGR2RGB),
            (512, 512),
        )
        target = cv2.resize(
            cv2.imread(self.target_path[idx], cv2.IMREAD_GRAYSCALE), (512, 512)
        )

        target = np.where(target > 0, 255, 0)

        if self.transform is not None:
            aug = self.transform(image=img, target=target)
            img = Image.fromarray(aug["image"])
            target = aug["target"]

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])

        if self.test is False:
            img = t(img)

        target = dense_target(target)
        target = torch.from_numpy(target).long()

        return img, target


if __name__ == "__main__":
    pass
