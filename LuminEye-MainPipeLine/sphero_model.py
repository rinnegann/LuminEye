import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import hpe
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
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N = 15  # Initial Number of Frames

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = torch.load("/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/U2net/u2net_multiclass_epoch_200_batch_2/Miche_model_2023_01_09_23:22:49_val_iou0.907.pt")

n_classes = 3
batch_size = 1

colors = [[0,   0,   0], [0, 255, 0], [0, 0, 255]]
label_colours = dict(zip(range(n_classes), colors))

valid_classes = [0, 85, 170]
class_names = ["Background", "Pupil", "Iris"]


class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)



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



def decode_segmap(temp):
    # convert gray scale to color
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def find_contours(image, height, width, top_left, resize_amt=512):
    pred_image = decode_segmap(image) * 255

    pred_image = pred_image.astype(np.uint8)

    edge_detected_image = cv2.Canny(pred_image, 0, 200)

    contours, hierarchy = cv2.findContours(
        edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # finding contour with #maximum area
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # Relative to the Image Frame
    cx = int((cx/resize_amt) * width) + top_left[0]

    cy = int((cy/resize_amt) * height) + top_left[1]

    # minValOfIris = (findMinEyeX(cnt)/resize_amt) * width + top_left[0]
    
    minValOfIris = findMinEyeX(cnt)
    
    minValOfIris =  ((minValOfIris/resize_amt) * width) + top_left[0]

    radius_of_iris = cx-minValOfIris

    eyeBallRadius = (12.2/5.9) * radius_of_iris

    return cx, cy, eyeBallRadius


def rect_to_bb(rect):
    # we will take the bounding box predicted by dlib library
    # and convert it into (x, y, w, h) where x, y are coordinates
    # and w, h are width and height
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)  # ( 68,2)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)  # [0,0]--> (x,y)

    # return the list of (x, y)-coordinates
    return coords


def findSymX(shape_arr):
    x_coord = []

    for (x, y) in shape_arr[27:31]:

        x_coord.append(x)

    x_coord.append(shape_arr[33][0])
    x_coord.append(shape_arr[51][0])
    x_coord.append(shape_arr[62][0])
    x_coord.append(shape_arr[66][0])
    x_coord.append(shape_arr[57][0])
    x_coord.append(shape_arr[8][0])

    return sum(x_coord)/len(x_coord)


def findMinEyeX(contour):
    minVal = contour[0][0][0]
    for i in contour:
        if i[0][0] < minVal:
            minVal = i[0][0]

    return minVal


def transformData(right_eye_image, left_eye_image, resize_amt):

    transform = A.Compose([
        A.Resize(resize_amt, resize_amt),
        A.augmentations.transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    aug_r = transform(image=right_eye_image)

    aug_l = transform(image=left_eye_image)

    l_img = aug_l["image"]
    r_img = aug_r["image"]

    return l_img, r_img


def predict_image_mask(model, image):
    model.eval()

    image = image.to(device)

    with torch.no_grad():

        softmax = nn.Softmax(dim=1)
        image = image.unsqueeze(0)

        model_output, _, _, _, _, _, _ = model(image)

        output = softmax(model_output)

        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


def captureFaceLandmarks(image):

    # Face Bounding Box
    rects = detector(image, 1)[0]  # --> (x,y,w,h)

    x, y, w, h = rect_to_bb(rects)

    if w * h > 2000:

        shape = predictor(image, rects)
        shape_arr = shape_to_np(shape)

    return shape_arr


def findInnerOuterEyeCorners(shape_arr):
    left_inner_eye_corner = shape_arr[39]
    right_inner_eye_corner = shape_arr[42]

    left_outer_eye_corner = shape_arr[36]
    right_outer_eye_corner = shape_arr[45]

    return left_inner_eye_corner, left_outer_eye_corner, right_inner_eye_corner, right_outer_eye_corner


def calculateSpheroParameters(image,frameCounter):

    shape_arr = captureFaceLandmarks(image)

    meanWidth = shape_arr[16][0] - shape_arr[0][0]
    
    
    R = meanWidth / 2

    symmAxis_x = findSymX(shape_arr=shape_arr)

    piyot_y = shape_arr[33][1]

    l_iris_center_x, l_iris_center_y, r_iris_center_x, r_iris_center_y, l_eye_ball_radius, r_eye_ball_radius, left_inner_eye_corner, left_outer_eye_corner, right_inner_eye_corner, right_outer_eye_corner = prediction(
        image, shape_arr)
    
    
    r_l = l_eye_ball_radius
    r_r = r_eye_ball_radius

    # Compute the rotation matrix from the measurement matrix
    r = hpe.compute_rotation(shape_arr)

    # Compute the head rotation angles from the rotation matrix
    yaw, pitch, roll, yaw_deg, pitch_deg, roll_deg = hpe.compute_angles(
        r, frameCounter)

    theta = yaw
    phi = pitch
    
    ######
    innerEyeCornerRot_right_x = cornerX(innerEyeCorner_x=right_inner_eye_corner[0], symmAxis_x=symmAxis_x , R=R, theta=theta)
    innerEyeCornerRot_right_y = cornerY(innerEyeCorner_y=right_inner_eye_corner[1], pivot_y=piyot_y, phi=phi)
    
    
    irisRot_right_x = irisX(innerEyeCorner_x=right_inner_eye_corner[0], outerEyeCorner_x=right_outer_eye_corner[0], symmAxis_x=symmAxis_x, r=r_r, R=R, iris_x=r_iris_center_x, theta=theta) 
    
    irisRot_right_y = irisY(innerEyeCorner_y=right_inner_eye_corner[1], outerEyeCorner_y=right_outer_eye_corner[1], pivot_y=piyot_y, r=r_r, iris_y=r_iris_center_y, phi=phi)
    
    #####
    
    innerEyeCornerRot_left_x = cornerX(innerEyeCorner_x=left_inner_eye_corner[0], symmAxis_x=symmAxis_x, R=R, theta=theta)
    innerEyeCornerRot_left_y = cornerY(innerEyeCorner_y=left_inner_eye_corner[0], pivot_y=piyot_y, phi=phi)
    
    irisRot_left_x = irisX(innerEyeCorner_x=left_inner_eye_corner[0], outerEyeCorner_x=left_outer_eye_corner[0], symmAxis_x=symmAxis_x, r=r_l, R=R, iris_x=l_iris_center_x, theta=theta)
    irisRot_left_y = irisY(innerEyeCorner_y=left_inner_eye_corner[1], outerEyeCorner_y=left_outer_eye_corner[1], pivot_y=piyot_y, r=r_l, iris_y=l_iris_center_x, phi=phi)
    
    #######
    

    return


def prediction(image, shape_arr):
    Leye = {"top_left": shape_arr[18], "bottom_right": shape_arr[28]}

    Reye = {"top_left": [shape_arr[28][0], shape_arr[25][1]],
            "bottom_right": [shape_arr[25][0], shape_arr[28][1]]}

    left_eye = image[Leye["top_left"][1]:Leye["bottom_right"]
                     [1], Leye["top_left"][0]:Leye["bottom_right"][0]]

    right_eye = image[Reye["top_left"][1]:Reye["bottom_right"]
                      [1], Reye["top_left"][0]:Reye["bottom_right"][0]]

    left_eye_image = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
    right_eye_image = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)

    l_eye, r_eye = transformData(
        right_eye_image, left_eye_image, resize_amt=512)

    pred_l_eye = predict_image_mask(model, l_eye)
    pred_r_eye = predict_image_mask(model, r_eye)

    l_iris_center_x, l_iris_center_y, l_eye_ball_radius = find_contours(
        pred_l_eye, height=left_eye_image.shape[0], width=left_eye_image.shape[1], top_left=Leye["top_left"], resize_amt=512)
    r_iris_center_x, r_iris_center_y, r_eye_ball_radius = find_contours(
        pred_r_eye, height=right_eye_image.shape[0], width=right_eye_image.shape[1], top_left=Reye["top_left"], resize_amt=512)

    left_inner_eye_corner, left_outer_eye_corner, right_inner_eye_corner, right_outer_eye_corner = findInnerOuterEyeCorners(
        shape_arr)

    return l_iris_center_x, l_iris_center_y, r_iris_center_x, r_iris_center_y, l_eye_ball_radius, r_eye_ball_radius, left_inner_eye_corner, left_outer_eye_corner, right_inner_eye_corner, right_outer_eye_corner






# define a video capture object
vid = cv2.VideoCapture(0)


frameCounter = 0


while (True):

    ret, frame = vid.read()

    frameCounter += 1
    l_x1, l_y1, r_x1, r_y1 = calculateSpheroParameters(frame, frameCounter)

    # cv2.circle(frame, (l_x1, l_y1), 1, (0, 0, 255), -1)
    # cv2.circle(frame, (r_x1, r_y1), 1, (0, 255, 0), -1)
    # print(x1,y1,x2,y2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
