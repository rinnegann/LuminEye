# from realesrgan.archs.srvgg_arch import SRVGGNetCompact
# from realesrgan import RealESRGANer
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassPrecision
import shutup
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
# import segmentation_models_pytorch as smp
import torch.nn.functional as F
import cv2
import time

import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from matplotlib import pyplot as plt

import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shutup.please()


sys.path.append(
    "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-MainPipeLine")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


eps = 1e-10

model = torch.load(
    "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/U2NET_MULTICLASS_IMG_256_DIC_batch_8/Miche_model_2023_04_11_22:14:26_val_iou0.900.pt", map_location=device)


# model = torch.load("/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/U2NET_MULTICLASS_IMG_256_DIC_batch_8/Miche_model_2023_04_11_22:14:26_val_iou0.900.pt")

#
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

colors = [[0,   0,   0], [0, 255, 0], [0, 0, 255]]
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


# model_gan = SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
#                             num_feat=64, num_conv=16, upscale=2, act_type='prelu')


# upsampler = RealESRGANer(
#     scale=netscale,
#     model_path=gan_model_path,
#     dni_weight=dni_weight,
#     model=model_gan,
#     tile=tile,
#     tile_pad=tile_pad,
#     pre_pad=pre_pad,
#     half=not fp32,
#     gpu_id=gpu_id)


def decode_segmap(temp):
    # convert gray scale to color
    # temp=temp.numpy()
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


def calculate_e1(gt, pred):
    class_labels = [0, 1, 2]

    E1 = 0
    for class_idx in class_labels:

        img_all = np.equal(gt, class_idx)

        pred_all = np.equal(pred, class_idx)

        # tp = np.dot(img_all,pred_all).sum()

        e1 = (np.logical_xor(img_all, pred_all).sum()) / \
            (pred.shape[0]*pred.shape[1])

        E1 += e1

    return E1/len(class_labels)


valid_x = sorted(
    glob(f"{val_images}/*"))
valid_y = sorted(glob(f"{val_masks }/*"))


def IoU(pred, true_pred, smooth=1e-10, n_classes=n_classes):
    with torch.no_grad():
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        pred = pred.contiguous().view(-1)
        true_pred = true_pred.contiguous().view(-1)

        iou_class = []
        for value in range(0, n_classes):
            true_class = pred == value
            true_label = true_pred == value

            if true_label.long().sum().item() == 0:
                iou_class.append(np.nan)

            else:

                inter = torch.logical_and(
                    true_class, true_label).sum().float().item()
                union = torch.logical_or(
                    true_class, true_label).sum().float().item()

                iou = (inter + smooth)/(union + smooth)
                iou_class.append(iou)

        return np.nanmean(iou_class)


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


class Iris(Dataset):
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
            img = aug['image']
            mask = aug['mask']
        return img, mask, (h, w), self.images[index].split("/")[-1]


def get_images(test_x, test_y, val_transform, batch_size=1, shuffle=True, pin_memory=True):

    val_data = Iris(test_x, test_y, transform=val_transform, enhance=False)
    test_batch = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, drop_last=True)
    return test_batch


val_transform = A.Compose([
    A.Resize(img_resize, img_resize),
    A.augmentations.transforms.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_batch = get_images(valid_x, valid_y, val_transform, batch_size=batch_size)

val_cls = Iris(valid_x, valid_y, transform=val_transform, enhance=False)


def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)

    edge_detected_image = cv2.Canny(median, 0, 200)

    contours, hierarchy = cv2.findContours(
        edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def mean_iou(image, mask):

    metric_precision = MulticlassPrecision(num_classes=3)
    metric_recall = MulticlassRecall(num_classes=3)

    image = image.to(device)
    mask = mask.to(device)

    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0)

    unnorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    with torch.no_grad():

        softmax = nn.Softmax(dim=1)

        model_output, _, _, _, _, _, _ = model(image)

        predicted_label = F.softmax(model_output, dim=1)
        predicted_label = torch.argmax(predicted_label, dim=1)

        # Predicted Mask
        pred_mask = predicted_label.permute(
            1, 2, 0).squeeze(-1).detach().cpu().numpy()

        # GT Mask

        gt_mask = mask.permute(1, 2, 0).squeeze(-1).detach().cpu().numpy()

        pred_mask = decode_segmap(pred_mask) * 255.0

        gt_mask = decode_segmap(gt_mask) * 255.0

        gt_grayscale = np.argmax(gt_mask, axis=-1)
        pred_grayscale = np.argmax(pred_mask, axis=-1)

        # print(f"Prediction Shape:- {gt_grayscale .shape}")

        # print(np.unique( np.argmax(pred_mask,axis=-1)))
        print(f"Prediction Shape:- {gt_grayscale.shape}")
        precision_score = metric_precision(torch.from_numpy(
            pred_grayscale), torch.from_numpy(gt_grayscale))

        recall_score = metric_recall(torch.from_numpy(
            pred_grayscale), torch.from_numpy(gt_grayscale))

        print(f"Precision:-{precision_score} | Recall:- {recall_score} ")

        # nice_e1_score = np.logical_xor(pred_mask,gt_mask).sum()/ (pred_mask.shape[0] * pred_mask.shape[1])

        nice_e1_score = calculate_e1(pred=pred_grayscale, gt=gt_grayscale)

        img = unnorm(image).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        predicted_label = predicted_label.contiguous().view(-1)  # 65536

        mask = mask.contiguous().view(-1)  # 65536

        iou_single_class = []

        for class_member in range(0, n_classes):
            # print(class_member)
            true_predicted_class = predicted_label == class_member
            true_label = mask == class_member

            if true_label.long().sum().item() == 0:
                iou_single_class.append(np.nan)

            else:
                intersection = (torch.logical_and(
                    true_predicted_class,
                    true_label
                ).sum().float().item())

                union = (torch.logical_or(
                    true_predicted_class,
                    true_label
                ).sum().float().item())

                iou = (intersection + eps)/(union + eps)

                iou_single_class.append(iou)

    return iou_single_class, img, gt_mask, pred_mask, nice_e1_score, precision_score, recall_score


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
        
        print(f'Image shape: {image.shape}')

        iou_class, img, gt_mask, pred_mask, nice_e1_score, precision_score, recall_score = mean_iou(
            image=image, mask=mask)

        PRECISION += precision_score
        RECALL += recall_score

        F1_SCORE += (2*precision_score * recall_score) / \
            (precision_score+recall_score)

        E1_SCORE += nice_e1_score

        pred_contours = find_contours(pred_mask.astype(np.uint8))
        gt_contours = find_contours(gt_mask.astype(np.uint8))

        for pred_cnt in pred_contours:  # Green for Prediction
            cv2.drawContours(img, [pred_cnt],  -1, (0, 255, 0), 1)

        for gt_cnt in gt_contours:  # Blue for Ground Truth
            cv2.drawContours(img, [gt_cnt],  -1, (0, 0, 255), 1)

        bg_iou, pupil_iou, iris_iou = iou_class

        # print(bg_iou,pupil_iou,iris_iou)
        
        print(image.shape)
        white_space = np.full((256, 10, 3), 255, dtype=np.uint8)
        combined_image = np.hstack((img[:,:,::-1]*255.0, white_space, gt_mask, white_space, pred_mask))
        
        cv2.imwrite(os.path.join(saved_location,image_name),combined_image)

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

            fig.savefig(os.path.join(saved_location,image_name))
            # plt.show()
            plt.tight_layout()
            plt.close()

        total_bg += bg_iou
        total_pupil += pupil_iou
        total_iris += iris_iou

        # if i ==1:
        #     break

    end_time = time.time()

    print(f"{end_time-start_time} Taken to complete the predictions")
    return total_bg/len(val_batch), total_pupil/len(val_batch), total_iris/len(val_batch), E1_SCORE/len(val_batch), PRECISION / len(val_batch), RECALL/len(val_batch), F1_SCORE/len(val_batch)


if __name__ == "__main__":

    saved_location = "Predictions/FinalPredictions"
    total_bg, total_pupil, total_iris, e1_score, precision, recall, f1_score = main(
        saved_location, visualize=False)

    print(f"E1 Score:- {e1_score} ")
    print(f1_score)
