import cv2
from glob import glob
import os
import numpy as np

image_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Padded/val_image"
mask_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Padded/val_masks"
save_location = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Padded/val_multi_classes"

images = sorted(glob(f"{image_path}/*"))
masks = sorted(glob(f"{mask_path}/*"))

# print(images)
# print(masks)

valid_classes = [0,85, 170]
class_names = ["Background","Pupil","Iris"]


class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)


colors = [ [  0,   0,   0],[0,255,0],[0,0,255]]
label_colours = dict(zip(range(n_classes), colors))


def encode_segmap(mask):
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


def getting_boundary_from_mask(images,masks,save_location):
    
    if not os.path.exists(save_location):
        os.makedirs(save_location)
        
    count = 0
    color_map = [[0,0,0],[0,255,0],[0,0,255]]
    
    for x,y in zip(images,masks):
        
        print(x)
        print(y)
        image = cv2.imread(x)
        
        raw_image = cv2.imread(y)
        raw_image = cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)
        
        ori_image = image.copy()
        
        
        
        gray = cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 5)
        
        bg_mask = np.zeros_like(raw_image)


        contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        max_contour = 0
        for contour in contours:
            # approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            area = cv2.contourArea(contour)
            
            
            if area > max_contour:
                cv2.drawContours(bg_mask,[contour],0,(0,0,255),-1)
            else:
                cv2.drawContours(bg_mask,[contour],0,(0,255,0),-1)
            
    
            max_contour = area
                
        output_mask = []
        for i,color in enumerate(color_map):
            cmap = np.all(np.equal(bg_mask,color),axis=-1)
            output_mask.append(cmap)
        output_mask = np.stack(output_mask,axis=-1)

        grayscale_mask = np.argmax(output_mask, axis=-1)
        grayscale_mask = (grayscale_mask / len(color_map)) * 255
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)
        grayscale_mask = encode_segmap(grayscale_mask)
        
        
                        
        cv2.imwrite(os.path.join(save_location,x.split("/")[-1].split(".")[0]+".png"),grayscale_mask)
        
        

                
if __name__ == "__main__":
    getting_boundary_from_mask(images, masks, save_location)
