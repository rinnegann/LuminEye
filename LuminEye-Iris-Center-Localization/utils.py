import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A


def load_model(model_path):
    """Load Regression model

    Args:
        model_path (_str_): _model path_


    Returns:
        _torch model_: _RESNET model_
    """

    model = torch.load(model_path, map_location=device)

    model.eval()

    return model


def selectImgDir(datasetType: str):

    if datasetType == "MP2GAZE":
        return "/home/nipun/Documents/Uni_Malta/Datasets/"

    elif datasetType == "i2head":
        return "/home/nipun/Documents/Uni_Malta/Datasets/"

    elif datasetType == "GI4E":

        return "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/images/"


def captureFaceLandmarks(frame):

    results = face_mesh.process(frame)
    landmarks = results.multi_face_landmarks[0]

    shape_arr = []

    for landmark in landmarks.landmark:

        x = landmark.x
        y = landmark.y

        relative_x = int(x * frame.shape[1])
        relative_y = int(y * frame.shape[0])

        shape_arr.append([relative_x, relative_y])

    return np.array(shape_arr)


def cropped_image(img, shape_array, padded_amt):
    """Cropped eye region

    Args:
        img (__numpy__): _Original Image_
        shape_array (_numpy_): _FaceLandMark locations_
        padded_amt (int, optional): _padding size_. Defaults to 15.

    """

    Leye = {"top_left": shape_array[70], "bottom_right": shape_array[133]}

    Reye = {"top_left": shape_array[285], "bottom_right": shape_array[263]}

    left_eye = img[
        Leye["top_left"][1] : Leye["bottom_right"][1] + padded_amt,
        Leye["top_left"][0] : Leye["bottom_right"][0],
    ]

    right_eye = img[
        Reye["top_left"][1] : Reye["bottom_right"][1] + padded_amt,
        Reye["top_left"][0] : Reye["bottom_right"][0],
    ]

    return left_eye, right_eye, Leye, Reye


def rescale_coordinate(coord, original_image, resize_amt):

    h, w = original_image.shape[:2]
    coord[0] = int((coord[0] / resize_amt) * w)
    coord[1] = int((coord[1] / resize_amt) * h)

    return coord


def mpArrayToNumpy(landmark_array, img):

    shape_arr = []

    for landmark in landmark_array.landmark:
        x = landmark.x
        y = landmark.y

        relative_x = int(x * img.shape[1])
        relative_y = int(y * img.shape[0])

        shape_arr.append([relative_x, relative_y])

    return np.array(shape_arr)


def scaleCoorinatesToOriginalImage(pred_coords, eye_margin):

    # {'top_left': array([385, 214]), 'bottom_right': array([426, 226])}

    x1 = eye_margin["top_left"][0] + pred_coords[0]
    y1 = eye_margin["top_left"][1] + pred_coords[1]

    return [x1, y1]


def calculateEuclideanDistance(coord1, coord2):
    return (
        ((coord1[0]) - float(coord2[0])) ^ 2 + (float(coord1[0]) - float(coord2[0])) ^ 2
    ) ^ 0.5


def CheckForLess(list1, val):

    l1 = []
    for x in list1:

        if x <= val:
            l1.append(x)
    return l1


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


def heatmap2argmax(heatmap, scale=False):
    N, C, H, W = heatmap.shape
    index = heatmap.view(N, C, 1, -1).argmax(dim=-1)
    pts = torch.cat([index % W, index // W], dim=2)

    if scale:
        scale = torch.Tensor([W, H], device=heatmap.device)
        pts = _scale(pts, scale)

    return pts


def _scale(p, s):
    return 2 * (p / s) - 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prediction_imageV1(model, image, resize_amt):

    model.eval()

    img = image[:, :, ::-1]

    img = cv2.resize(img, (resize_amt, resize_amt))

    img = img / 255.0

    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        out_coord = heatmap2argmax(model(img).view(-1, 1, 256, 256))

    img = img.squeeze(0)

    image = transforms.ToPILImage()

    pred_coord = out_coord.detach().cpu().numpy()[0][0]

    return image, pred_coord


def prediction_imageV2(model, image, resize_amt):

    val_transforms = A.Compose(
        [
            A.Resize(width=resize_amt, height=resize_amt),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1),
        ]
    )

    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transformed_img = val_transforms(image=image[:, :, ::-1])
    image = transformed_img["image"]

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        out_coord = model(image)

    image = image.squeeze(0)

    image = transforms.ToPILImage()(unnorm(image))

    pred_coord = out_coord.detach().cpu().numpy()[0]

    return image, pred_coord


def imgTransform(image, resize_amt):
    """Transform numpy images to Torch tensors

    Args:
        image (_numpy_): _cropped_eye_region_
        resize_amt (_int_): _Input Resize for the image_

    Returns:
        _torch.float32_: _Transformed image tensor_
    """

    transform = A.Compose(
        [
            A.Resize(resize_amt, resize_amt),
            A.augmentations.transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
            ToTensorV2(),
        ]
    )

    return transform(image=image)["image"]


def predict_image_masku2net(model, image, resize_amt):

    image = imgTransform(image[:, :, ::-1], resize_amt)

    image = image.to(device)

    with torch.no_grad():

        softmax = nn.Softmax(dim=1)
        image = image.unsqueeze(0)

        model_output, _, _, _, _, _, _ = model(image)

        output = softmax(model_output)

        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


def decode_segmap(temp, n_classes=3):

    colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255]]
    label_colours = dict(zip(range(n_classes), colors))
    # convert gray scale to color
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def imageCenterFromConnectedComponent(img, connectivity=4):

    pred_image = decode_segmap(img) * 255

    grayscale_mask = np.argmax(pred_image, axis=-1).astype(np.uint8)

    analysis = cv2.connectedComponentsWithStats(
        grayscale_mask, connectivity, cv2.CV_32S
    )
    (totalLabels, label_ids, values, centroid) = analysis

    return centroid


def findCentersMoments(pred_mask, eye_w, eye_h, margin):

    pred_image = decode_segmap(pred_mask) * 255
    pred_image = pred_image.astype(np.uint8)
    edge_detected_image = cv2.Canny(pred_image, 0, 200)

    contours, hierarchy = cv2.findContours(
        edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    cnt = max(contours, key=cv2.contourArea)  # finding contour with #maximum area
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    ori_cx = int(margin["top_left"][0] + ((cx / RESIZE_AMT) * eye_w))
    ori_cy = int(margin["top_left"][1] + ((cy / RESIZE_AMT) * eye_h))

    return [ori_cx, ori_cy]



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss
