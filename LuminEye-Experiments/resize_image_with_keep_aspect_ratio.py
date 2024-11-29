"""Resize only one dimension while keeping the other dimension equal to the aspect ratio
   (Here Width will be resized and height changes according to the aspect ratio)
"""


import os 
import cv2
import numpy as np
from glob import glob
from PIL import Image
import PIL
from PIL import Image



image_path = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/train_img"
mask_path = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Dataset/train_masks"

saved_location =  "./Images_with_Aspect_Ratio"

if not os.path.exists(saved_location):
    os.makedirs(saved_location)

images = sorted(glob(f"{image_path}/*"))
masks = sorted(glob(f"{mask_path}/*"))

image_resize = 128


def resize_aspect_ratio(image_name,resize_width,source,splt):
    img = Image.open(image_name)
    
    img_name = image_name.split("/")[-1]  # 300 # 400
    print(f"Image Name:{img_name}")
    wpercent = (resize_width/float(img.size[0]))  # 512/300 --> 1.70
    hsize = int((float(img.size[1])*float(wpercent))) # 680
    
    img = img.resize((resize_width,hsize), PIL.Image.ANTIALIAS)
    
    if source =="image":
        
        # print(os.path.join(saved_location,f"{splt}_{source}"))
        if not os.path.exists(os.path.join(saved_location,f"{splt}_{source}")):
            os.makedirs(os.path.join(saved_location,f"{splt}_{source}"))
            
    else:
        # print(os.path.join(saved_location,f"{splt}_{source}"))
        if not os.path.exists(os.path.join(saved_location,f"{splt}_{source}")):
            os.makedirs(os.path.join(saved_location,f"{splt}_{source}"))
            
    
    img.save(os.path.join(saved_location,f"{splt}_{source}",img_name))


def main():
    
    count = 0 
    for x,y in zip(images,masks):
    
        
        
        img = resize_aspect_ratio(x, image_resize,"image","train")
        mask = resize_aspect_ratio(y,image_resize,"masks","train")
        
        # if count == 0:
        #     break
        
    print("Images & Masks have been Resized")
    

if __name__ == "__main__":
    main()
        
        
        
    
    

    
    
    