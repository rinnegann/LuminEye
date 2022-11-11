import cv2
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label,regionprops,find_contours


class BinaryMaskToBBOX:
    def __init__(self,dataset_path,img_dir,msk_dir,save_location,ext):
        self.dataset_path = dataset_path
        
        
        self.img_dir = img_dir
        self.msk_dir= msk_dir
        
        self.ext = ext
        self.save_location = os.path.join(self.dataset_path,save_location)
        
        
        
        self.create_dir(self.save_location)
    
    def create_dir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
            
    def mask_to_border(self,mask):
        h,w = mask.shape[:2]
        
        border = np.zeros((h,w))
        
        contours = find_contours(mask,128)
        
        for contour in contours:
            for c in contour:
                x = int(c[0])
                y = int(c[1])
                border[x][y] = 255
        return border
    
    def mask_to_bbox(self,mask):
        bboxes = []
        mask = self.mask_to_border(mask)
        props = regionprops(mask.astype(int))
        
        for prop in props:
            
            x1 = int(prop.bbox[1])
            y1 = int(prop.bbox[0])

            x2 = int(prop.bbox[3])
            y2 = int(prop.bbox[2])

            bboxes.append([x1, y1, x2, y2])

        return bboxes
    def parse_mask(self,mask):
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)
        return mask
    
    def  __getitem__(self,amt):
        
        images_sample = random.sample(sorted(glob.glob(os.path.join(
            self.dataset_path, self.img_dir) + "/**", recursive=True)),amt)
        
        for idx,x in enumerate(images_sample):
            
            file_name = x.split("/")[-1].split(".")[0]+"."+self.ext
            
            img = cv2.imread(x,cv2.IMREAD_COLOR)
            
            # print(os.path.join(self.dataset_path,self.msk_dir,file_name))
            msk   = cv2.imread(os.path.join(self.dataset_path,self.msk_dir,file_name),cv2.IMREAD_GRAYSCALE)
            
            
            bboxes = self.mask_to_bbox(msk)
    
            for bbox in bboxes:
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    
            # Contacatenate Images
            cat_image = np.concatenate([img,self.parse_mask(msk)],axis=1)
            
            
            
            cv2.imwrite(os.path.join(self.save_location,x.split("/")[-1]), cat_image)
            

            
if __name__ == "__main__":
    
    IMG_DIR = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/UBRIS/"
    SAVE_DIR = "visualization"
    ext = "tiff"
    BinaryMaskToBBOX(IMG_DIR,"train_img","train_masks",SAVE_DIR,ext)[4]
    
    """IMG_DIR = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/"
    SAVE_DIR = "visualization"
    ext = "bmp"
    BinaryMaskToBBOX(IMG_DIR,"img","inv_masks",SAVE_DIR,ext)[4]"""