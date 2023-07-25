import os
import cv2
import numpy as np
import torch
import dlib
from imutils import face_utils
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from torchvision import transforms  
import torch.nn as nn
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn.functional as F
import math
import time
from imutils import face_utils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import mediapipe
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


detector = None
predictor = None
GAN_MODEL = None
IRIS_MODEL = None
EYE_AR_THRESH = 0.2

mp_face_mesh = mediapipe.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

RESIZE_AMT = 64


def prediction_image(model,image):
        
        val_transforms =  A.Compose([
                                        A.Resize(width=RESIZE_AMT,height=RESIZE_AMT),
                                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ToTensorV2(p=1)
                                        ])
        
        unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transformed_img = val_transforms(image=image[:,:,::-1])
        image = transformed_img['image']
        
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
                out_coord = model(image)
        
        
        image = image.squeeze(0)

        image = transforms.ToPILImage()(unnorm(image))
        
        
        pred_coord = out_coord.detach().cpu().numpy()[0]
        
        return image,pred_coord






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


def load_model(model_path):
    """Load Regression model

    Args:
        model_path (_str_): _model path_
    

    Returns:
        _torch model_: _RESNET model_
    """

    model = torch.load(model_path,map_location=device)

    model.eval()

    return model




class BB_model(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = efficientnet_b3(pretrained=True)
        
        layers = list(efficientnet.children())[:1]
        self.features1 = nn.Sequential(*layers)

    
        self.bb = nn.Sequential(nn.BatchNorm1d(1536),nn.Linear(1536,512),nn.ReLU(inplace=True),
                                nn.BatchNorm1d(512),nn.Linear(512,2))
        
    def forward(self,x):
        x = self.features1(x) #[1, 1536, 8, 8]
        x = F.relu(x)
        
        
        x = nn.AdaptiveAvgPool2d((1,1))(x) # [ 1,1536,1,1]
        
        
        x = x.view(x.shape[0],-1) # [1,1536]

        
        
        return self.bb(x)





# class BB_model(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         resnet = models.resnet34(weights=True)
        
#         # for param in resnet.parameters():
            
#         #     param.requires_grad = False
        
#         layers = list(resnet.children())[:8]
#         self.features1 = nn.Sequential(*layers[:6])
#         self.features2  = nn.Sequential(*layers[6:])
    
#         self.bb = nn.Sequential(
#                                 nn.Linear(512,256),
#                                 nn.BatchNorm1d(256),
#                                 nn.ReLU(inplace=True),
#                                 nn.Linear(256,128),
#                                 nn.BatchNorm1d(128),
#                                 nn.ReLU(inplace=True),
#                                 nn.Linear(128,2),
#                                 )
        
#     def forward(self,x):
#         x = self.features1(x) # 1, 128, 32, 32
        
#         x = self.features2(x) # [1, 512, 8, 8]
        
#         x = F.relu(x)
        
        
#         x = nn.AdaptiveAvgPool2d((1,1))(x) # [ 1,512,1,1]
        
        
#         x = x.view(x.shape[0],-1) # [1, 512]
        
#         return self.bb(x)


def cropped_image(img, shape_array, padded_amt=30):
    """Cropped eye region 

    Args:
        img (__numpy__): _Original Image_
        shape_array (_numpy_): _FaceLandMark locations_
        padded_amt (int, optional): _padding size_. Defaults to 15.

    """

    Leye = {"top_left": shape_array[70], "bottom_right": shape_array[133]}

    Reye = {"top_left": shape_array[285],
            "bottom_right": shape_array[263]}

    left_eye = img[Leye["top_left"][1]:Leye["bottom_right"][1] +
                   15
                   , Leye["top_left"][0]:Leye["bottom_right"][0]]

    right_eye = img[Reye["top_left"][1]:Reye["bottom_right"][1] +
                    15, Reye["top_left"][0]:Reye["bottom_right"][0]]
    
    
    
    # Reye['top_left'][0] = Reye['top_left'][0] - 5

    return left_eye, right_eye, Leye, Reye


def rescale_coordinate(coord,original_image,resize_amt):
    
    h,w = original_image.shape[:2]
    coord[0] = int((coord[0]/resize_amt) * w)
    coord[1] = int((coord[1]/resize_amt) * h)
    
    return coord

def main(visualize_iris=True,enhance=True):
    
    
    vid = cv2.VideoCapture(0)
    
    frameCounter = 0
    
    while True:

        ret, frame = vid.read()
        
        
        
        # MediaPipe
        shape_array = captureFaceLandmarks(frame)
        
        
        left_eye, right_eye,Leye,Reye = cropped_image(frame, shape_array)
        
        

            
        _,pred_l_eye = prediction_image(model=REGRESSION_MODEL,image=left_eye)
        
        
        _,pred_r_eye = prediction_image(model=REGRESSION_MODEL,image=right_eye)
        
        
        pred_l_eye = rescale_coordinate(pred_l_eye,left_eye,RESIZE_AMT)
        
        pred_r_eye = rescale_coordinate(pred_r_eye,right_eye,RESIZE_AMT)
        
        
        
        
        
        cv2.circle(left_eye,(int(pred_l_eye[0]),int(pred_l_eye[1])),1,(0,255,0),-1)
        cv2.circle(right_eye,(int(pred_r_eye[0]),int(pred_r_eye[1])),1,(0,255,0),-1)
        
        
        cv2.imshow("Frame", frame)
        
        
        left_eye = cv2.resize(left_eye,(512,512))
        right_eye = cv2.resize(right_eye,(512,512))
        cv2.imshow("Right Eye",right_eye)
        cv2.imshow("Left Eye",left_eye)
        

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        
    vid.release()
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    
    # Mix Dataset
    # regression_model_path = '/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Iris-Center-Localization/Regression_model_1.487574208665777.pth'
    
    
    # H2DATASET WITH Efficientenet
    
    regression_model_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Iris-Center-Localization/Regression_EfficientNet__epoch_200_mae_summation_batch_64_resize_64_only_h2dataset/Regression_model_1.4015370882474458.pth"
    

    REGRESSION_MODEL = load_model(
        model_path=regression_model_path)
    

    main()