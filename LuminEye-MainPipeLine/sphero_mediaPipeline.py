import os
import cv2
import numpy as np
import torch
import dlib
from imutils import face_utils
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import torch.nn as nn
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import hpe
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




def findSymX(shape_arr):
    """Return average symmetric axis x coordinate

    Args:
        shape_arr (_np.numpy_): _FaceLandMark Coordinates_

    Returns:
        _float32_: _x coordinate of symmetric x axis_
    """
    x_cord =  []
    coordinates_of_symmetric = [9,8,
                            168,
                            6,
                            197,
                            195,
                            5,
                            4,
                            1,
                            19,
                            94,
                            2,
                            164,
                            0,
                            11,
                            12,
                            13,
                            14,
                            15, 16, 17, 18]
    
    
    for num in coordinates_of_symmetric:
        x_cord.append(shape_arr[num][0])
    
    
    return sum(x_cord)/len(x_cord)


def calculateParameters(shape_array):
    """Return meanWidth,piyotY and SymmX along with Radius of the face

    Args:
        shape_array (_numpy_): _FaceLandMark locations_
    """
    
    

    meanWidth = shape_array[356][0] - shape_array[127][0]

    symmAxis_x = findSymX(shape_arr=shape_array)

    piyot_y = shape_array[4][1]
    
        

    return meanWidth, symmAxis_x, piyot_y, meanWidth/2



        
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



def cropped_image(img, shape_array, padded_amt=15, enhance=True):
    """Cropped eye region and enhance it according to the requirement

    Args:
        img (__numpy__): _Original Image_
        shape_array (_numpy_): _FaceLandMark locations_
        padded_amt (int, optional): _padding size_. Defaults to 20.
        enhance (bool, optional): _applying SISR_. Defaults to True.
    """

    Leye = {"top_left": shape_array[70], "bottom_right": shape_array[133]}

    Reye = {"top_left": shape_array[285] ,
            "bottom_right": shape_array[263]}

    left_eye = img[Leye["top_left"][1]:Leye["bottom_right"][1] +
                   padded_amt, Leye["top_left"][0]:Leye["bottom_right"][0]]

    right_eye = img[Reye["top_left"][1]:Reye["bottom_right"][1] +
                    padded_amt, Reye["top_left"][0]:Reye["bottom_right"][0]]

    if enhance:
        left_eye, _ = GAN_MODEL.enhance(left_eye)
        right_eye, _ = GAN_MODEL.enhance(right_eye)

    return left_eye, right_eye,Leye,Reye


def imgTransform(image):
    """ Transform numpy images to Torch tensors

    Args:
        image (_numpy_): _cropped_eye_region_
        resize_amt (_int_): _Input Resize for the image_

    Returns:
        _torch.float32_: _Transformed image tensor_
    """

    transform = A.Compose([
        A.Resize(RESIZE_AMT, RESIZE_AMT),
        A.augmentations.transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    return transform(image=image)["image"]

def predict_image_masku2net(model, image):
    
    image = imgTransform(image[:,:,::-1])

    image = image.to(device)


    with torch.no_grad():

        softmax = nn.Softmax(dim=1)
        image = image.unsqueeze(0)

        model_output, _, _, _, _, _, _ = model(image)

        output = softmax(model_output)

        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


def load_model(model_path, model_input_size):
    """Load Segmentation model

    Args:
        model_path (_str_): _model path_
        model_input_size (_int_): _Input size of the model_

    Returns:
        _torch model_: _U2NET model_
    """

    model = torch.load(model_path,map_location=device)

    model.eval()

    return model, model_input_size

def decode_segmap(temp,n_classes=3):
    
    
    colors = [ [  0,   0,   0],[0,255,0],[0,0,255]]
    label_colours = dict(zip(range(n_classes), colors))
    #convert gray scale to color
    temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def drawContours(frame,pred_mask,h,w,margin):
    """Draw Iris and Pupil Contours on Original Image

    Args:
        frame (_numpy_): _original image frame_
        pred_mask (_torch.float32_): _prediction from model_
        h (_int_): _cropped eye image height_
        w (_int_): _cropped eye image height_
        margin (_dic_): _Eye Top Left x and y coordinates
        
        
    """
    
    
    pred_image = decode_segmap(pred_mask) * 255
    pred_image = pred_image.astype(np.uint8)
    edge_detected_image = cv2.Canny(pred_image, 0, 200)
    
    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = {}
    
    for contour in contours:
    
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        # print(area)
        if ((len(approx) > 8) & (len(approx) < 23) & (area > 30) ):
            
            
            max_area[area] = contour
            
    x = sorted(max_area,key = lambda x:x)


    print(x)
    max_contour = max_area[x[-1]]
    min_contour = max_area[x[1]]
    

    for coords in min_contour:
    
        coords[0][0] = margin["top_left"][0]+ ((coords[0][0]/RESIZE_AMT) * w)
        coords[0][1] = margin["top_left"][1] + ((coords[0][1]/RESIZE_AMT) * h)
        

    for coords in max_contour:
    
        coords[0][0] = margin["top_left"][0]  + ((coords[0][0]/RESIZE_AMT) * w)
        coords[0][1] = margin["top_left"][1] + ((coords[0][1]/RESIZE_AMT) * h)
    
    cv2.drawContours(frame,[max_contour],0,(0,0,255),1)
    cv2.drawContours(frame,[min_contour],0,(0,255,0),1)
    
    return frame


def findMinEyeX(cntour):
    minVal = cntour[0][0][0]
    for i in cntour:
        if i[0][0] < minVal:
            minVal = i[0][0]
            
    return minVal


def findRadiusIris(pred_mask,eye_w,eye_h,margin):
    
    pred_image = decode_segmap(pred_mask) * 255
    pred_image = pred_image.astype(np.uint8)
    edge_detected_image = cv2.Canny(pred_image, 0, 200)
    
    contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    
    cnt = max(contours, key = cv2.contourArea) # finding contour with #maximum area
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    
    ori_cx = int(margin["top_left"][0] +((cx/RESIZE_AMT ) * eye_w))
    ori_cy = int(margin["top_left"][1] + ((cy/RESIZE_AMT ) * eye_h))
    
    
    
    minValOfIris = findMinEyeX(cnt)
    
    
    minValOfIris = margin["top_left"][0] +((minValOfIris/RESIZE_AMT ) * eye_w)
    
    
    
    radius_of_iris = ori_cx-minValOfIris
    
    
    eyeBallRadius = (12.2/5.9) * radius_of_iris
    
    
    return eyeBallRadius,ori_cx,ori_cy

    
    
    
    
    
    

# Compute the rotated iris centre x-coordinate, for each eye
def irisX(innerEyeCorner_x, outerEyeCorner_x, symmAxis_x, r, R, iris_x, theta):
    
    # Calculate the centre point of the eyeball with respect to the symmetry axis of the face
    eyeMid_x = ((innerEyeCorner_x + outerEyeCorner_x) / 2) - symmAxis_x
    eyeMid_y = r - R
    
    # Calculate the radius, Re
    Re = math.sqrt(eyeMid_x ** 2 + eyeMid_y ** 2)
    
    # Calculate the angle, omega_h
    wh = math.asin(eyeMid_x / Re)
    
    # Calculate the angle, omega_e
    we = math.asin(((iris_x - symmAxis_x) - eyeMid_x) / r)
    
    # Calculate the new centre point of the eyeball with respect to the symmetry axis of the
    # face, after a head rotation by the yaw angle, theta
    eyeMidRot_x = Re * math.sin(wh + math.radians(theta))
    
    # Calculate the new image position of the iris centre after rotating by the head yaw angle
    irisRot_x = (r * math.sin(we + math.radians(theta))) + eyeMidRot_x

    return irisRot_x


# Compute the rotated iris centre y-coordinate, for each eye
def irisY(innerEyeCorner_y, outerEyeCorner_y, pivot_y, r, iris_y, phi):
    
    # Calculate the centre point of the eyeball with respect to a pivot point
    eyeMid_y = ((innerEyeCorner_y + outerEyeCorner_y) / 2) - pivot_y
    
    # Calculate the angle between the projection of the point on the eyeball sphere and the
    # eyeball centre
    
    # print(eyeMid_y)
    
    # print(iris_y)
    
    # print(pivot_y)
    
    # print(r)
    
    # print((eyeMid_y - (iris_y - pivot_y)) / r)
    we = math.asin((eyeMid_y - (iris_y - pivot_y)) / r)
    
    # Calculate the new image position of the iris centre after rotating by the head pitch angle
    irisRot_y = (eyeMid_y * math.cos(math.radians(phi))) - (r * math.sin(we + math.radians(phi)))
    
    return irisRot_y




# Compute the rotated inner eye corner x-coordinate, for each eye
def cornerX(innerEyeCorner_x, symmAxis_x, R, theta):
    
    # Calculate the eye corner position with respect to the symmetry axis of the face
    x = innerEyeCorner_x - symmAxis_x
    
    # Calculate the angle between the eye corner and the symmetry axis of the face
    wc = math.asin(x / R)
    
    # Calculate the new image position of the eye corner after rotating by the head yaw angle
    innerEyeCornerRot_x = R * math.sin(wc + math.radians(theta))
    
    return innerEyeCornerRot_x


# Compute the rotated inner eye corner y-coordinate, for each eye
def cornerY(innerEyeCorner_y, pivot_y, phi):
    
    # Calculate the eye corner position with respect to a pivot point
    y = innerEyeCorner_y - pivot_y
    
    # Calculate the new image position of the eye corner after rotating by the head pitch angle
    innerEyeCornerRot_y = y * math.cos(math.radians(phi))
    
    return innerEyeCornerRot_y




def initialization_function(frame,frameCounter,visualize=True,enhance=False):
    
    # MediaPipe
    shape_array = captureFaceLandmarks(frame)
    
    meanWidth, symmAxis_x, piyot_y, R = calculateParameters(shape_array)
    
    left_eye, right_eye,Leye,Reye = cropped_image(frame, shape_array,enhance=False)
    
    # Prediction from the model for both eyes
    pred_l_eye,pred_r_eye = predict_image_masku2net(IRIS_MODEL,left_eye),predict_image_masku2net(IRIS_MODEL, right_eye)
    
    
    print(f"Right Eye shape:- {right_eye.shape}")
    print(f"Left Eye shape:- {left_eye.shape}")
    
    if visualize:
            
                try:
                    frame = drawContours(frame,pred_l_eye,h=left_eye.shape[0]/2 if enhance else left_eye.shape[0],w=left_eye.shape[1]/2 if enhance else left_eye.shape[1],margin=Leye)
                    frame = drawContours(frame,pred_r_eye,h=right_eye.shape[0]/2 if enhance else right_eye.shape[0],w=(right_eye.shape[1]/2) if enhance else right_eye.shape[1],margin=Reye)
                except IndexError:
                    pass
    
    r = hpe.compute_rotation(shape_array)
        
    theta, phi, roll, yaw_deg, pitch_deg, roll_deg = hpe.compute_angles(r, frameCounter)
    
    
    
    # Inner Iris X and Y coordinate of Right eye
    innerEyeCorner_right_x = shape_array[362][0]
    innerEyeCorner_right_y = shape_array[362][1]

    # Outer Iris X and Y coordinate of Right eye
    outerEyeCorner_right_x = shape_array[263][0]
    outerEyeCorner_right_y = shape_array[263][1]

    # Inner Iris X and Y coordinate of Left eye
    innerEyeCorner_left_x = shape_array[133][0]
    innerEyeCorner_left_y = shape_array[133][1]

    # Outer Iris X and Y coordinate of Left eye
    outerEyeCorner_left_x = shape_array[33][0]
    outerEyeCorner_left_y = shape_array[33][1]
    
    
    innerEyeCornerRot_right_x = cornerX(innerEyeCorner_x=innerEyeCorner_right_x, symmAxis_x=symmAxis_x, R=R, theta=theta)
    innerEyeCornerRot_right_y = cornerY(innerEyeCorner_y=innerEyeCorner_right_y, pivot_y=piyot_y, phi=phi)

    innerEyeCornerRot_left_x = cornerX(innerEyeCorner_x=innerEyeCorner_left_x, symmAxis_x=symmAxis_x, R=R, theta=theta)
    innerEyeCornerRot_left_y = cornerY(innerEyeCorner_y=innerEyeCorner_left_y, pivot_y=piyot_y, phi=phi)
    
    
    
    l_rad_iris,l_cx,l_cy=findRadiusIris(pred_l_eye,eye_w=left_eye.shape[1],eye_h=left_eye.shape[0],margin=Leye)
    r_rad_iris,r_cx,r_cy=findRadiusIris(pred_r_eye,eye_w=right_eye.shape[1],eye_h=right_eye.shape[0],margin=Reye)
    
    
    
    l_rad_iris,l_cx,l_cy=findRadiusIris(pred_l_eye,eye_w=left_eye.shape[1],eye_h=left_eye.shape[0],margin=Leye)
    r_rad_iris,r_cx,r_cy=findRadiusIris(pred_r_eye,eye_w=right_eye.shape[1],eye_h=right_eye.shape[0],margin=Reye)
    

    print(f"Left Eye Radius: {l_rad_iris}")
    
    print(f"Right Eye Radius: {r_rad_iris}")
        
    # Fixed Eye Radius for Initial Stage
    
    l_rad_iris = 10
    r_rad_iris = 10
    
    # # Rotation For Right Eye
    irisRot_right_x = irisX(innerEyeCorner_x=innerEyeCorner_right_x, outerEyeCorner_x=outerEyeCorner_right_x, 
                            symmAxis_x=symmAxis_x, r=r_rad_iris, R=R, iris_x=r_cx, theta=theta)
    
    irisRot_right_y = irisY(innerEyeCorner_y=innerEyeCorner_right_y, outerEyeCorner_y=outerEyeCorner_right_y, 
                            pivot_y=piyot_y, r=r_rad_iris, iris_y=r_cy, phi=phi)



    # Rotation for Left Eye

    irisRot_left_x = irisX(innerEyeCorner_x=innerEyeCorner_left_x, outerEyeCorner_x=outerEyeCorner_left_x,
                        symmAxis_x=symmAxis_x, r=l_rad_iris, R=R, iris_x=l_cx, theta=theta)

    irisRot_left_y = irisY(innerEyeCorner_y=innerEyeCorner_left_y, outerEyeCorner_y=outerEyeCorner_left_y, 
                        pivot_y=piyot_y, r=l_rad_iris, iris_y=l_cy, phi=phi)
    
    
    
    return left_eye,right_eye




def main():
    
    
    vid = cv2.VideoCapture(0)
    
    frameCounter = 0
    
    init_frame= 0 
    
    while True:

        ret, frame = vid.read()
        
        
        while init_frame<15:
            
            
            left_eye,right_eye=initialization_function(frame,init_frame)
            
            
            left_eye = cv2.resize(left_eye,(256,256))
            right_eye = cv2.resize(right_eye,(256,256))
            cv2.imshow("Right Eye",right_eye)
            cv2.imshow("Left Eye",left_eye)
                
            
            init_frame += 1
            
        
    
        frameCounter += 1
        # print(frameCounter)
        
        cv2.imshow("Frame", frame)
        
        
        # left_eye = cv2.resize(left_eye,(512,512))
        # right_eye = cv2.resize(right_eye,(512,512))
        # cv2.imshow("Right Eye",right_eye)
        # cv2.imshow("Left Eye",left_eye)
        

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        
    vid.release()
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    
    
    eye_segmentation_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/U2NET_MULTICLASS_IMG_256_DIC_batch_8/Miche_model_2023_04_11_22:14:26_val_iou0.900.pt" 
    
    
    
    IRIS_MODEL, RESIZE_AMT = load_model(
        model_path=eye_segmentation_path, model_input_size=256)
    
    
    
    main()