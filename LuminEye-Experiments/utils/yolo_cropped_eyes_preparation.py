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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/train_img"
train_masks = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/train_masks"


val_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/val_img"

val_masks = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/val_masks"


train_x = sorted(
    glob(f"{train_images}/*"))
train_y = sorted(
    glob(f"{train_masks}/*"))
valid_x = sorted(
    glob(f"{val_images}/*"))
valid_y = sorted(
    glob(f"{val_masks }/*"))


yolo_model_path = "/home/nipun/Music/yolov5/runs/train/exp8/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=yolo_model_path, force_reload=True)

# Filter BBOX Based on Confidence
model.conf = 0.40


saved_img_location = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/YoloCroppedEyes/images"
saved_mask_location = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/YoloCroppedEyes/masks"


if not os.path.exists(saved_img_location):
    os.makedirs(saved_img_location)


if not os.path.exists(saved_mask_location):
    os.makedirs(saved_mask_location)


def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)

    edge_detected_image = cv2.Canny(median, 0, 200)

    contours, hierarchy = cv2.findContours(
        edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def main(images, masks, visualize=False):

    RESIZE_AMT = 256

    for z, (img_path, mask_path) in enumerate(zip(valid_x, valid_y)):

        pad_top_x1 = 50
        pad_bottom_y2 = 50
        pad_top_left_y1 = 30
        pad_bottom_right_x2 = 15

        image = cv2.imread(img_path)

        h, w = image.shape[:2]

        mask = cv2.imread(mask_path)

        print(img_path)

        results = model(image[:, :, ::-1], size=640)

        df = results.pandas().xyxy[0]

        # Get the all BBOX related to Iris Class which is zero
        df = df[df["class"] == 0]

        df = df[df['confidence'] == df['confidence'].max()]

        for (i, row) in df.iterrows():

            x1 = round(row["xmin"])
            y1 = round(row["ymin"])
            x2 = round(row["xmax"])
            y2 = round(row["ymax"])
        print(image.shape)
        print(x1, y1, x2, y2)
        copy_img = image.copy()

        if (y1 - pad_top_left_y1) < 0:
            pad_top_left_y1 = y1

        if (x1-pad_top_x1) < 0:
            pad_top_x1 = x1

        cropped_img = copy_img[y1-pad_top_left_y1:y2 +
                               pad_bottom_y2, x1-pad_top_x1:x2+pad_bottom_right_x2]

        cropped_mask = mask[y1-pad_top_left_y1:y2 +
                            pad_bottom_y2, x1-pad_top_x1:x2+pad_bottom_right_x2]
        cropped_mask = np.where(cropped_mask > 0, 255, 0)

        image_name = img_path.split("/")[-1]
        mask_name = mask_path.split("/")[-1]

        # draw_img = cropped_img.copy()

        # gt_contours  = find_contours(cropped_mask.astype(np.uint8))

        # for gt_cnt in gt_contours:
        #         cv2.drawContours(cropped_img, [gt_cnt],  -1, (0,255,0), 1)

        # for gt_cnt in gt_contours:
        #     cv2.drawContours(draw_img, [gt_cnt],  -1, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(saved_img_location, image_name), cropped_img)
        cv2.imwrite(os.path.join(
            saved_mask_location, mask_name), cropped_mask)

        # plt.imshow(cropped_img[:,:,::-1])
        # plt.show()

        # plt.imshow(cropped_mask)
        # plt.show()

        # if z==50:
        #     break


if __name__ == "__main__":
    main(valid_x, valid_y)
