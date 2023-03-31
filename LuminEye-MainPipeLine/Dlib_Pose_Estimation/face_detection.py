import cv2
import os
import numpy as np
import time
import dlib
import imutils


modelPath = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-MainPipeLine/Dlib_Pose_Estimation/face_model/mmod_human_face_detector.dat"

def convert_and_trim_bb(image, rect,mode="CNN"):
    
    if mode == "CNN":
        for rct in rect:
        
            startX = rct.rect.left()
            startY = rct.rect.top()
            endX = rct.rect.right()
            endY = rct.rect.bottom()
    else:
	
        for rct in rect:
            
            startX = rct.left()
            startY = rct.top()
            endX = rct.right()
            endY = rct.bottom()
	
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
	# return our bounding box coordinates
    return (startX, startY, w, h)


def ChoseDetector(mode,modelPath=modelPath):
    
    if mode =="CNN":
        detector = dlib.cnn_face_detection_model_v1(modelPath)
        return detector
    else:
        detector = dlib.get_frontal_face_detector()
        return detector
    
def main(img_path):
    image = cv2.imread(img_path)

    image = cv2.resize(image[:,:,::-1],(600,600))

    detector = ChoseDetector(mode="CNN")
    
    rects = detector(image)
    
    
    boxes = [convert_and_trim_bb(image, rects,mode="CNN")]
    
    for (x, y, w, h) in boxes:
	
	    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
    cv2.imshow("Output", image[:,:,::-1])
    cv2.waitKey(0)
     
    
        
        
if __name__ == "__main__":
    imgPath = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/images/003_01.png"
    main(imgPath)
    













