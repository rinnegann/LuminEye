"""Used this file to get Boundary on Ground Truth and evaluate ground truth masks boundary with it's corresponding
masks"""

import cv2
from glob import glob
import os
import numpy as np

image_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Padded/val_image"
mask_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Padded/val_masks"
save_location = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Padded/Boundaries"

images = sorted(glob(f"{image_path}/*"))
masks = sorted(glob(f"{mask_path}/*"))

# print(images)
# print(masks)


def getting_boundary_from_mask(images,masks,save_location):
    
    if not os.path.exists(save_location):
        os.makedirs(save_location)
        
    count = 0
    
    for x,y in zip(images,masks):
        
        print(x)
        print(y)
        image = cv2.imread(x)
        
        raw_image = cv2.imread(y)
        
        ori_image = image.copy()
        
        
        
        gray = cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 5)


        contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            area = cv2.contourArea(contour)
            print(area)
            if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
                contour_list.append(contour)

        cv2.drawContours(image, contour_list,  -1, (0,255,0), 1)
        vertical_bar = np.ones((ori_image.shape[0],100,3),np.uint8)
        
        
        # print(f"Original Image {ori_image.shape}")
        # print(f"Boundary Imagae {image.shape}")
        # print(f"Mask Image {raw_image.shape}")

        final_image = np.concatenate((ori_image,vertical_bar*255,image,vertical_bar*255,raw_image),axis = 1)
        cv2.imwrite(os.path.join(save_location,x.split("/")[-1].split(".")[0]+".png"),final_image)
        
        # count +=1
        # if count ==1:
        #     break
        

                
if __name__ == "__main__":
    getting_boundary_from_mask(images, masks, save_location)
