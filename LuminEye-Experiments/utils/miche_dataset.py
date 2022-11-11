import cv2
import os
import numpy as np
from rich.progress import track


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def inverse_img(MASK_PATH,SAVE_LOCATION):
    for f in track(os.scandir(MASK_PATH),total=len(os.listdir(MASK_PATH))):
        if f.is_file():
            msk = cv2.imread(f.path)
            inv_mask = cv2.bitwise_not(msk)
            
            create_dir(SAVE_LOCATION)
            
            cv2.imwrite(os.path.join(SAVE_LOCATION,f.name), inv_mask)
        
        


if __name__ == "__main__":
    msk_path = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/masks/"
        
    save_location = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/inv_masks"
    
    inverse_img(msk_path,save_location)   