import shutup
import torch
import torch.nn as nn
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
from glob import glob
import cv2
import time
from matplotlib import pyplot as plt
import sys

sys.path.insert(1, "../")
from utils import *
from dataset_classes import IrisV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shutup.please()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


eps = 1e-10

model = torch.load(
    "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/U2NET_MULTICLASS_IMG_256_DIC_batch_8/Miche_model_2023_04_11_22:14:26_val_iou0.900.pt",
    map_location=device,
)

# model = torch.load("/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/U2NET_MULTICLASS_IMG_256_DIC_batch_8/Miche_model_2023_04_11_22:14:26_val_iou0.900.pt")

# model = torch.load("/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/u2net_multiclass_epoch_200_batch_2_with_dice_and_boundary_loss/Miche_model_2023_01_17_20:33:14_val_iou0.906.pt")


# 512 Boundary Loss
# model = torch.load("/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/u2net_multiclass_epoch_200_batch_2_with_dice_and_boundary_loss/Miche_model_2023_01_17_20:33:14_val_iou0.906.pt")

# UBris v2 Model
# model = torch.load(
#     "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/Miche_model_2023_09_27_14:18:07_val_iou0.899.pt")

val_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/val_img"

val_masks = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/val_masks/"
n_classes = 3
batch_size = 1

img_resize = 256

colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255]]
label_colours = dict(zip(range(n_classes), colors))

valid_classes = [0, 85, 170]
class_names = ["Background", "Pupil", "Iris"]

gan_model_path = "/home/nipun/Music/Real-ESRGAN/experiments/net_g_latest.pth"
dni_weight = None
tile = 0
tile_pad = 10
pre_pad = 0
fp32 = True
gpu_id = 0
netscale = 2


class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)


"""model_gan = SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                            num_feat=64, num_conv=16, upscale=2, act_type='prelu')


upsampler = RealESRGANer(
    scale=netscale,
    model_path=gan_model_path,
    dni_weight=dni_weight,
    model=model_gan,
    tile=tile,
    tile_pad=tile_pad,
    pre_pad=pre_pad,
    half=not fp32,
    gpu_id=gpu_id)"""


valid_x = sorted(glob(f"{val_images}/*"))
valid_y = sorted(glob(f"{val_masks }/*"))


def get_images(
    test_x, test_y, val_transform, batch_size=1, shuffle=True, pin_memory=True
):

    val_data = IrisV1(test_x, test_y, transform=val_transform, enhance=False)
    test_batch = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return test_batch


val_transform = A.Compose(
    [
        A.Resize(img_resize, img_resize),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)

val_batch = get_images(valid_x, valid_y, val_transform, batch_size=batch_size)

val_cls = IrisV1(valid_x, valid_y, transform=val_transform, enhance=False)


def main(saved_location=None, visualize=False):

    if not os.path.exists(saved_location):
        os.makedirs(saved_location)

    # if not os.path.exists(saved_location):
    #     os.makedirs(saved_location)

    total_bg = 0
    total_pupil = 0
    total_iris = 0
    # if not os.path.exists(saved_location):
    #     os.makedirs(saved_location)
    unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    total_iou = 0

    E1_SCORE = 0

    PRECISION = 0
    RECALL = 0

    F1_SCORE = 0

    start_time = time.time()
    for i in range(len(val_batch)):
        image, mask, (h, w), image_name = val_cls[i]

        print(f"Image shape: {image.shape}")

        (
            iou_class,
            img,
            gt_mask,
            pred_mask,
            nice_e1_score,
            precision_score,
            recall_score,
        ) = mean_iou(image=image, mask=mask)

        PRECISION += precision_score
        RECALL += recall_score

        F1_SCORE += (2 * precision_score * recall_score) / (
            precision_score + recall_score
        )

        E1_SCORE += nice_e1_score

        pred_contours = find_contours(pred_mask.astype(np.uint8))
        gt_contours = find_contours(gt_mask.astype(np.uint8))

        for pred_cnt in pred_contours:  # Green for Prediction
            cv2.drawContours(img, [pred_cnt], -1, (0, 255, 0), 1)

        for gt_cnt in gt_contours:  # Blue for Ground Truth
            cv2.drawContours(img, [gt_cnt], -1, (0, 0, 255), 1)

        bg_iou, pupil_iou, iris_iou = iou_class

        # print(bg_iou,pupil_iou,iris_iou)

        print(image.shape)
        white_space = np.full((256, 10, 3), 255, dtype=np.uint8)
        combined_image = np.hstack(
            (img[:, :, ::-1] * 255.0, white_space, gt_mask, white_space, pred_mask)
        )

        cv2.imwrite(os.path.join(saved_location, image_name), combined_image)

        if visualize:

            fig, axes = plt.subplots(1, 3, figsize=(10, 10))
            axes[0].set_title("Original Image")
            axes[0].text(3, 300, f"Height: {h} | Width: {w}")
            axes[1].set_title("Ground Truth Image")
            axes[2].set_title("Predicted Image")

            axes[0].axis("off")
            axes[1].axis("off")
            axes[2].axis("off")

            axes[0].imshow(img)
            axes[1].imshow(gt_mask)
            axes[2].imshow(pred_mask)

            # plt.close('all')

            fig.savefig(os.path.join(saved_location, image_name))
            # plt.show()
            plt.tight_layout()
            plt.close()

        total_bg += bg_iou
        total_pupil += pupil_iou
        total_iris += iris_iou

    end_time = time.time()

    print(f"{end_time-start_time} Taken to complete the predictions")
    return (
        total_bg / len(val_batch),
        total_pupil / len(val_batch),
        total_iris / len(val_batch),
        E1_SCORE / len(val_batch),
        PRECISION / len(val_batch),
        RECALL / len(val_batch),
        F1_SCORE / len(val_batch),
    )


if __name__ == "__main__":

    saved_location = "Predictions/FinalPredictions"
    total_bg, total_pupil, total_iris, e1_score, precision, recall, f1_score = main(
        saved_location, visualize=False
    )

    print(f"E1 Score:- {e1_score} ")
    print(f1_score)
