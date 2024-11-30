import torch
import numpy as np
import cv2


class CenterDatasetHM(torch.utils.data.Dataset):
    def __init__(self, df, image_dir=IMAGE_DIR, RESIZE_AMT=64):

        self.RESIZE_AMT = RESIZE_AMT
        self.image_dir = image_dir
        self.df = df
        self.image_ids = df.Image_Name.unique()
        self.transforms = transforms

    # apply gaussian kernel to image
    def _gaussian(self, xL, yL, sigma, H, W):

        channel = [
            math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma**2))
            for r in range(H)
            for c in range(W)
        ]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel

    # convert original image to heatmap
    def _convertToHM(self, img, keypoints, sigma=2):

        H = img.shape[0]
        W = img.shape[1]
        nKeypoints = len(keypoints)

        img_hm = np.zeros(shape=(H, W, nKeypoints // 2), dtype=np.float32)

        for i in range(0, nKeypoints // 2):
            x = keypoints[i * 2]
            y = keypoints[1 + 2 * i]

            channel_hm = self._gaussian(x, y, sigma, H, W)

            img_hm[:, :, i] = channel_hm

        return img_hm

    def __getitem__(self, ix):

        img_id = self.image_ids[ix]
        img_path = os.path.join(self.image_dir, img_id)

        img = cv2.imread(img_path)[:, :, ::-1]

        img = cv2.resize(img, (self.RESIZE_AMT, self.RESIZE_AMT))

        img = img / 255.0

        data = self.df[self.df["Image_Name"] == img_id]

        x1 = data["X1"].values[0] * self.RESIZE_AMT
        y1 = data["Y1"].values[0] * self.RESIZE_AMT

        heatmap = (
            torch.tensor(self._convertToHM(img, [x1, y1]), dtype=torch.float32)
            .permute(2, 0, 1)
            .view(1 * self.RESIZE_AMT * self.RESIZE_AMT)
        )

        image = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        return image, heatmap

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.image_ids)


# Regression
############################################################################

class CenterDatasetRG(torch.utils.data.Dataset):
    def __init__(self, df, image_dir=IMAGE_DIR, transforms=None):
        self.image_dir = image_dir
        self.df = df
        self.image_ids = df.ImageName.unique()
        self.transforms = transforms

    def __getitem__(self, ix):

        img_id = self.image_ids[ix]
        img_path = os.path.join(self.image_dir, img_id)

        img = cv2.imread(img_path)[:, :, ::-1]

        data = self.df[self.df["ImageName"] == img_id]

        x1 = data["X1"].values[0] * RESIZE_AMT
        y1 = data["Y1"].values[0] * RESIZE_AMT

        center_loc = torch.Tensor([x1, y1]).float()

        if self.transforms:
            transformed = self.transforms(image=img)

            image = transformed["image"]

        return image, center_loc

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.image_ids)
