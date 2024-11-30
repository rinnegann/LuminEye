"""Used this file to get boundary around Ground Truth and Prediction on image itself"""


import cv2
from glob import glob
import os
import numpy as np

image_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Boundary_Loss/Binary_Segemnetation_with_SDF_BY_only_focussing_iris_region_batch_4_epoch_200_boundary_loss/experiment_no_1/img"
pred_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Boundary_Loss/Binary_Segemnetation_with_SDF_BY_only_focussing_iris_region_batch_4_epoch_200_boundary_loss/experiment_no_1/pred"
gt_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Boundary_Loss/Binary_Segemnetation_with_SDF_BY_only_focussing_iris_region_batch_4_epoch_200_boundary_loss/experiment_no_1/mask"
save_location = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Boundary_Loss/Binary_Segemnetation_with_SDF_BY_only_focussing_iris_region_batch_4_epoch_200_boundary_loss/experiment_no_1/pred_gt_boundaries"

images = sorted(glob(f"{image_path}/*"))
pred = sorted(glob(f"{pred_path}/*"))
gt = sorted(glob(f"{gt_path}/*"))

# print(images)
# print(masks)

def find_contours(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
        
    edge_detected_image = cv2.Canny(median, 0, 200)


    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


def getting_boundary_from_mask(images,pred,gt,save_location):
    
    if not os.path.exists(save_location):
        os.makedirs(save_location)
        
    count = 0
    
    for x,y,z in zip(images,pred,gt):
        
        print(x)
        print(y)
        image = cv2.imread(x)
        
        pred_image = cv2.imread(y)
        image = cv2.resize(image,(400,400))
        ori_image = image.copy()
        pred_image = cv2.resize(pred_image,(400,400))
        
        gt_image = cv2.imread(z)
        gt_image = cv2.resize(gt_image,(400,400))
        
        pred_contours = find_contours(pred_image)
        gt_contours = find_contours(gt_image)
        
        for pred_cnt in pred_contours:
            cv2.drawContours(image, [pred_cnt],  -1, (0,255,0), 1)
            
        for gt_cnt in gt_contours:
            cv2.drawContours(image, [gt_cnt],  -1, (0,0,255), 1)

        vertical_bar = np.ones((ori_image.shape[0],50,3),np.uint8)
        
        final_image = np.concatenate((ori_image,vertical_bar*255,image),axis = 1)

        
        cv2.imwrite(os.path.join(save_location,x.split("/")[-1].split(".")[0]+".png"),final_image)
        
        # count +=1
        # if count ==1:
        #     break
        

                
if __name__ == "__main__":
    getting_boundary_from_mask(images,pred,gt, save_location)
