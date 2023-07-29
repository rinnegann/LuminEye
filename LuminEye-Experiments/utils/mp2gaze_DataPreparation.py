# Dataclass Preparation
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
from torch.utils.data import Dataset
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
import mediapipe
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision import models


df = pd.read_csv("/home/nipun/Documents/Uni_Malta/Datasets/Center_Regression/MP2GAZE/mp2gaze_annotations.csv")


IMG_DIR = '/home/nipun/Documents/Uni_Malta/Datasets'


def draw_circle(img,array):
    
    for i in range(0,len(array),2):
    
        arr = array[i:i+2]
        
        
        cv2.circle(img,(arr[0],arr[1]),1,(0,255,0),1)
        
        
    return img


def cropeyes(image,inner_array,outer_array):
    
    arr = {"top_left":[inner_array[0]-5,inner_array[1]-20],"bottom_right":[outer_array[0]+5,outer_array[1]+20]}
    
    
    
    return img[arr["top_left"][1]:arr["bottom_right"][1],arr["top_left"][0]:arr["bottom_right"][0]],arr




def main(saved_dir):
    
    
    
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    
    cols = ["Image_Name", 'X1', 'Y1']
    image_count = 0
    data_array = []
    visualize = False
    for i,(_,rows) in enumerate(df.iterrows()):
        
        
    
        
        # print(rows)
        # 
        bbox =[int(float(x)) for x in rows["Coordinates"][1:-1].split(',') ] 
    
    
        left_inner= bbox[0:2]
        left_center = bbox[2:4]
        left_outer = bbox[4:6]
        
        
        right_inner = bbox[6:8]
        right_center = bbox[8:10]
        right_outer =  bbox[10:12]
        
        
        img_path = os.path.join(IMG_DIR,rows["ImageName"][:-1])
        # # 
        img = cv2.imread(img_path)
        

        print(rows["ImageName"][:-1])
        print("#"*10)
        print(img_path)
        
        
        # image = draw_circle(img.copy(),bbox)
        
        
        left_eye,Leye = cropeyes(img,left_inner,left_outer)
        right_eye,Reye = cropeyes(img,right_inner,right_outer)

        
        left_center[0] = left_center[0] - Leye["top_left"][0]
        left_center[1] = left_center [1] - Leye["top_left"][1]
        
        
        
        right_center[0] =right_center[0] - Reye["top_left"][0]
        right_center[1] = right_center[1] - Reye["top_left"][1]

        if visualize:
            
            
            cv2.circle(left_eye,(left_center[0],left_center[1]),1,(0,0,255),1)
            cv2.circle(right_eye,(right_center[0],right_center[1]),1,(0,0,255),1)
            
            fig, axs = plt.subplots(1, 2)

            axs[0].set_title("Left Eye")
            axs[1].set_title("Right Eye")

            axs[0].axis("off")
            axs[1].axis("off")

            axs[0].imshow(left_eye[:, :, ::-1])
            axs[1].imshow(right_eye[:, :, ::-1])


            plt.tight_layout()
            plt.show()
            plt.close('all')
        
        
        image_count +=1
        data_array.append({"Image_Name": f"{str(image_count)}_left.png",
                                        "X1": left_center[0]/left_eye.shape[1],
                                        "Y1": left_center[1]/left_eye.shape[0]})

        data_array.append({"Image_Name": f"{str(image_count)}_right.png",
                            "X1": right_center[0]/right_eye.shape[1],
                            "Y1": right_center[1]/right_eye.shape[0]})
        
        
        cv2.imwrite(os.path.join(
                            saved_dir, f"{str(image_count)}_left.png"), left_eye)

        cv2.imwrite(os.path.join(
            saved_dir, f"{str(image_count)}_right.png"), right_eye)

 
    df = pd.DataFrame(data_array, columns=cols)
    
    
    
    df.to_csv("mp2gaze_center.csv")
    
if __name__ == "__main__":
    saved_dir = 'mp2gazeImages'
    main(saved_dir)
