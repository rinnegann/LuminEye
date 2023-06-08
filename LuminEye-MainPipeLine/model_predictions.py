import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import hpe
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.optim import Adam
from tqdm import tqdm
from glob import glob
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import time
import mediapipe
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


GAN_MODEL = None
IRIS_MODEL = None


def irisGanModel(model_path, dni_weight=None, tile=0, tile_pad=10,
                 pre_pad=0, fp32=True, gpu_id=0, netscale=2):

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                            num_feat=64, num_conv=16, upscale=2, act_type='prelu')

    return RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)


def load_model(model_path, model_input_size):

    model = torch.load(model_path)

    model.eval()

    return model, model_input_size


def getLandMark_Coordinates(index, image, landmarks):

    landmark_index = landmarks.landmark[index]

    return (int(landmark_index.x * image.shape[1]), int(landmark_index.y * image.shape[0]))


def cropped_image(x1, y1, x2, y2, image, padded_amt=20, enhance=True):

    cropped_img = image[y1:y2+padded_amt, x1:x2]

    if enhance:
        upscale_img, _ = GAN_MODEL.enhance(cropped_img)

        return cropped_img, upscale_img

    return cropped_img


def save_img(image, image_name):
    cv2.imwrite(image_name, image)


def imgTransform(image, resize_amt):

    transform = A.Compose([
        A.Resize(resize_amt, resize_amt),
        A.augmentations.transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    return transform(image=image)["image"]


def draw_contours(image_tensor, orginal_eye, resize_amt):

    image = image_tensor.detach().cpu().numpy().astype(np.uint8)

    edge_detected_image = cv2.Canny(image, 0, 1)

    contours, hierarchy = cv2.findContours(
        edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    eye = cv2.resize(orginal_eye, (resize_amt, resize_amt))

    for pred_cnt in contours:
        cv2.drawContours(eye, [pred_cnt],  -1, (0, 255, 0), 1)

    return eye


def predict_image_masku2net(model, image):

    image = image.to(device)

    print(image.device)

    with torch.no_grad():

        softmax = nn.Softmax(dim=1)
        image = image.unsqueeze(0)

        model_output, _, _, _, _, _, _ = model(image)

        output = softmax(model_output)

        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked




def main(img_path,save=True):

    # Read Image
    img_base = cv2.imread(img_path)
    mp_face_mesh = mediapipe.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process(img_base)

    try:
        landmarks = results.multi_face_landmarks[0]



        lEyeTopLeftCorner = getLandMark_Coordinates(70, img_base, landmarks)

        rEyeTopLeftCorner = getLandMark_Coordinates(285, img_base, landmarks)

        lEyeBottomRight = getLandMark_Coordinates(133, img_base, landmarks)

        rEyeBottomRight = getLandMark_Coordinates(263, img_base, landmarks)

        l_eye, l_eye_gan = cropped_image(x1=lEyeTopLeftCorner[0], y1=lEyeTopLeftCorner[1],
                                        x2=lEyeBottomRight[0], y2=lEyeBottomRight[1], image=img_base)




        # save_img(l_eye,"Original_Leye.png")
        # save_img(l_eye_gan,"Original_Leye_gan.png")

        r_eye, r_eye_gan = cropped_image(x1=rEyeTopLeftCorner[0], y1=rEyeTopLeftCorner[1],
                                        x2=rEyeBottomRight[0], y2=rEyeBottomRight[1], image=img_base)



        # Eye Region WIthough Gan Enhancement
        l_img = imgTransform(l_eye[:,:,::-1], RESIZE_AMT)
        r_img = imgTransform(r_eye[:,:,::-1], RESIZE_AMT)

        # Eye Region WIth Gan Enhancement
        l_img_gan = imgTransform(l_eye_gan[:,:,::-1], RESIZE_AMT)
        r_img_gan = imgTransform(r_eye_gan[:,:,::-1], RESIZE_AMT)



        pred_mask_l = predict_image_masku2net(IRIS_MODEL, l_img)

        pred_mask_r = predict_image_masku2net(IRIS_MODEL, r_img)


        pred_mask_l_gan = predict_image_masku2net(IRIS_MODEL,l_img_gan)

        pred_mask_r_gan = predict_image_masku2net(IRIS_MODEL,r_img_gan)



        left_eye_boundary = draw_contours(pred_mask_l,l_eye,RESIZE_AMT)
        right_eye_boundary = draw_contours(pred_mask_r,r_eye,RESIZE_AMT)


        left_eye_boundary_gan = draw_contours(pred_mask_l_gan,l_eye_gan,RESIZE_AMT)
        right_eye_boundary_gan = draw_contours(pred_mask_r_gan,r_eye_gan,RESIZE_AMT) 


        if save:
            folder_name = img_path.split("/")[-1].split(".")[0]

            root_folder = f"{SAVED_LOCATION}/{folder_name}/{IMG_RESOLUTION}"

            if not os.path.exists(root_folder):
                os.makedirs(root_folder)
            cv2.imwrite(os.path.join(root_folder,"Original_Image.png"),img_base)

            save_img(img_base,os.path.join(root_folder,"Original_Image.png"))
            save_img(l_eye,os.path.join(root_folder,"Original_Leye.png"))
            save_img(l_eye_gan,os.path.join(root_folder,"Original_Leye_gan.png"))
            save_img(r_eye,os.path.join(root_folder,"Original_Reye.png"))
            save_img(r_eye_gan,os.path.join(root_folder,"Original_Reye_gan.png"))
            save_img(left_eye_boundary, os.path.join(root_folder,"left_eye_boundary.png"))
            save_img(right_eye_boundary,os.path.join(root_folder, "right_eye_boundary.png"))
            save_img(left_eye_boundary_gan , os.path.join(root_folder,"left_eye_gan_boundary.png"))
            save_img(right_eye_boundary_gan, os.path.join(root_folder,"right_eye_gan_boundary.png"))

    except Exception as e:
        print(f"This Error Has Raised {e}")



def batch_process(file_dir):
    
    for image_file in os.scandir(file_dir):
        if image_file.is_file():
            main(image_file.path)



if __name__ == "__main__":
    img_path = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/images/003_05.png"
    eye_modelgan_path = "/home/nipun/Music/Real-ESRGAN/experiments/net_g_30000.pth"
    eye_segmentation_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/u2net_multiclass_epoch_200_batch_2/Miche_model_2023_01_09_23:22:49_val_iou0.907.pt"

    file_dir = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/images"
    
    GAN_MODEL = irisGanModel(model_path=eye_modelgan_path)
    IRIS_MODEL, RESIZE_AMT = load_model(
        model_path=eye_segmentation_path, model_input_size=512)
    SAVED_LOCATION = "U2N2T_512_EYE_GAN"
    IMG_RESOLUTION = "NORMAL"
    
    # main(img_path)
    batch_process(file_dir=file_dir)
