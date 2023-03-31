# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


modelPath = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-MainPipeLine/Dlib_Pose_Estimation/face_model/shape_predictor_68_face_landmarks_GTX.dat"
imgPath = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/images/003_01.png"

# Face Detection
detector = dlib.get_frontal_face_detector()

# Face LandMark Detection
predictor = dlib.shape_predictor(modelPath)




# load the input image, resize it, and convert it to grayscale
image = cv2.imread(imgPath)
image = cv2.resize(image[:,:,::-1],(600,600))


# image = imutils.resize(image, width=500)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




rects = detector(image)



for (i,rect) in enumerate(rects):
    shape  = predictor(image,rect) # [(272, 134) (397, 259)]

    shape = face_utils.shape_to_np(shape)

    
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    for (x, y) in shape:    
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("LandMark Output",image[:,:,::-1])
cv2.waitKey(0)   


