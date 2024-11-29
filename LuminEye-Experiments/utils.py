import torch
from datetime import datetime
import numpy as np


def DiceBceLoss(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.to("cpu").squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = 1 - ((2.0 * intersection + eps) / (cardinality + eps)).mean()
    bce = F.cross_entropy(logits, true, reduction="mean")
    dice_bce = bce + dice_loss
    return dice_bce


def IoU(pred, true_pred, smooth=1e-10, n_classes=2):
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

                inter = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (inter + smooth) / (union + smooth)
                iou_class.append(iou)

        return np.nanmean(iou_class)


def pixel_wise_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())  # total number
    return accuracy


def dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)

    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx

    return dummy


def get_current_date_time():
    now = datetime.now()
    year = now.strftime("%Y")

    month = now.strftime("%m")

    day = now.strftime("%d")

    time = now.strftime("%H:%M:%S")

    return f"{year}_{month}_{day}_{time}_"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def visualize_multiclass_mask(train_batch):
    for img, mask in train_batch:
        print(img.shape)
        img1 = np.transpose(img[0, :, :, :], (1, 2, 0))
        mask1 = np.array(mask[0, :, :])
        img2 = np.transpose(img[1, :, :, :], (1, 2, 0))
        mask2 = np.array(mask[1, :, :])
        img3 = np.transpose(img[2, :, :, :], (1, 2, 0))
        mask3 = np.array(mask[2, :, :])
        fig, ax = plt.subplots(3, 2, figsize=(18, 18))
        ax[0][0].imshow(img1)
        ax[0][1].imshow(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].imshow(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].imshow(mask3)
        break


def calculate_e1(gt, pred):
    class_labels = [0, 1, 2]

    E1 = 0
    for class_idx in class_labels:

        img_all = np.equal(gt, class_idx)

        pred_all = np.equal(pred, class_idx)

        # tp = np.dot(img_all,pred_all).sum()

        e1 = (np.logical_xor(img_all, pred_all).sum()) / (pred.shape[0] * pred.shape[1])

        E1 += e1

    return E1 / len(class_labels)


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


def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)

    edge_detected_image = cv2.Canny(median, 0, 200)

    contours, hierarchy = cv2.findContours(
        edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

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
        pred_mask = predicted_label.permute(1, 2, 0).squeeze(-1).detach().cpu().numpy()

        # GT Mask

        gt_mask = mask.permute(1, 2, 0).squeeze(-1).detach().cpu().numpy()

        pred_mask = decode_segmap(pred_mask) * 255.0

        gt_mask = decode_segmap(gt_mask) * 255.0

        gt_grayscale = np.argmax(gt_mask, axis=-1)
        pred_grayscale = np.argmax(pred_mask, axis=-1)

        # print(f"Prediction Shape:- {gt_grayscale .shape}")

        # print(np.unique( np.argmax(pred_mask,axis=-1)))
        print(f"Prediction Shape:- {gt_grayscale.shape}")
        precision_score = metric_precision(
            torch.from_numpy(pred_grayscale), torch.from_numpy(gt_grayscale)
        )

        recall_score = metric_recall(
            torch.from_numpy(pred_grayscale), torch.from_numpy(gt_grayscale)
        )

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
                intersection = (
                    torch.logical_and(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item()
                )

                union = (
                    torch.logical_or(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item()
                )

                iou = (intersection + eps) / (union + eps)

                iou_single_class.append(iou)

    return (
        iou_single_class,
        img,
        gt_mask,
        pred_mask,
        nice_e1_score,
        precision_score,
        recall_score,
    )


if __name__ == "__main__":
    pass
