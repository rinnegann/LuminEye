"""
This code is responsible for following tasks
1. Get the cropped images of both right and left eye
2. Scale the Center coordinates according to the image size
3. Save images and center coornates as a csv file
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe
import pandas as pd
from glob import glob



mp_face_mesh = mediapipe.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def getCenterofEye(left_center, right_center, leye, reye):

    left_center = [float(left_center[0])-leye["top_left"]
                   [0], float(left_center[1])-leye["top_left"][1]]

    right_center = [float(right_center[0])-reye["top_left"]
                    [0], float(right_center[1])-reye["top_left"][1]]

    return left_center, right_center


def mpArrayToNumpy(landmark_array, img):

    shape_arr = []

    for landmark in landmark_array.landmark:
        x = landmark.x
        y = landmark.y

        relative_x = int(x * img.shape[1])
        relative_y = int(y * img.shape[0])

        shape_arr.append([relative_x, relative_y])

    return np.array(shape_arr)


def cropped_image(img, shape_array, padded_amt=15):
    """Cropped eye region 

    Args:
        img (__numpy__): _Original Image_
        shape_array (_numpy_): _FaceLandMark locations_
        padded_amt (int, optional): _padding size_. Defaults to 15.

    """

    Leye = {"top_left": shape_array[70], "bottom_right": shape_array[133]}

    Reye = {"top_left": shape_array[285],
            "bottom_right": shape_array[263]}

    left_eye = img[Leye["top_left"][1]:Leye["bottom_right"][1] +
                   padded_amt, Leye["top_left"][0]:Leye["bottom_right"][0]]

    right_eye = img[Reye["top_left"][1]:Reye["bottom_right"][1] +
                    padded_amt, Reye["top_left"][0]:Reye["bottom_right"][0]]

    return left_eye, right_eye, Leye, Reye


def mainAllCoords(text_dir, img_dir, csvType):
    data_array = []

    dataArrayAllCoordinates = []

    Allcols = ["ImageName", "Coordinates"]

    

    for text_path in text_dir:

        with open(text_path, "r") as f:
            info = [k.rstrip().split("\t") for k in f.readlines()]

       
        for obj in info:

            coords = []

            image_filepath = os.path.join(img_dir, obj[0])

            img = cv2.imread(image_filepath)
            
            print(image_filepath )

            results = face_mesh.process(img)

            if results.multi_face_landmarks is not None:
                landmarks = results.multi_face_landmarks[0]

                shape_arr = mpArrayToNumpy(landmarks, img)

                left_eye, right_eye, Leye, Reye = cropped_image(
                    img, shape_arr)

                bbox = [float(i) for i in obj[1:]]
             
                left_inner = bbox[6:8] 
                left_center = bbox[8:10]
                left_outer = bbox[10:12]

                right_inner = bbox[4:6]
                right_center = bbox[2:4]
                right_outer = bbox[0:2]
                coords.extend(left_outer)
                coords.extend(left_center)
                coords.extend(left_inner)
                
                
                
                
                
                coords.extend(right_inner)
                coords.extend(right_center)
                coords.extend(right_outer)

                dataArrayAllCoordinates.append({"ImageName": f"{obj[0].split('.')[0]}.png",
                                                "Coordinates": coords})

            else:
                print(
                    f"MediaPipe was failed to detect the faces on the image name {obj[0]}")
                continue
            
    print(f"Length of the Training Dataset{len(dataArrayAllCoordinates)}")
    
    
    

    allCoordinatesDF = pd.DataFrame(dataArrayAllCoordinates, columns=Allcols)
    
    
    # allCoordinatesDF = allCoordinatesDF.drop_duplicates(subset="ImageName")
    
    
    allCoordinatesDF.to_csv(f"gi4eAllCoordinates{csvType}.csv")
    
    return allCoordinatesDF


def mainCenterCsvCreation(dataframe, img_dir, saved_dir, csv_name):
    image_count = 0

    data_array = []

    cols = ["ImageName", "X1", "Y1"]

    for idx, row in dataframe.iterrows():

        img_path = os.path.join(img_dir, row["ImageName"])
        
        print(f"{row['Coordinates']}")
        
        print(row['Coordinates'][0])
        bbox =  row["Coordinates"]

        left_center = bbox[2:4]

        right_center = bbox[8:10]

        img = cv2.imread(img_path)

        results = face_mesh.process(img)

        if results.multi_face_landmarks is not None:

            image_count += 1

            img_name = row["ImageName"].split(".")[0]

            print(row["ImageName"])

            landmarks = results.multi_face_landmarks[0]

            shape_arr = mpArrayToNumpy(landmarks, img)

            left_eye, right_eye, Leye, Reye = cropped_image(
                img, shape_arr)

            left_center[0] = (float(left_center[0]) -
                              Leye['top_left'][0])/left_eye.shape[1]
            left_center[1] = (float(left_center[1]) -
                              Leye['top_left'][1])/left_eye.shape[0]
            right_center[0] = (float(right_center[0]) -
                               Reye['top_left'][0])/right_eye.shape[1]
            right_center[1] = (float(right_center[1]) -
                               Reye['top_left'][1])/right_eye.shape[0]

            if all(0 < k < 1 for k in left_center) and all(0 < l < 1 for l in right_center):
                
                image_count  +=1
                cv2.imwrite(os.path.join(
                    saved_dir, f"{img_name}_left.png"), left_eye)

                cv2.imwrite(os.path.join(
                    saved_dir, f"{img_name}_right.png"), right_eye)

                data_array.append({"ImageName": f"{img_name}_left.png",
                                   "X1": float(left_center[0]),
                                   "Y1": float(left_center[1])})

                data_array.append({"ImageName": f"{img_name}_right.png",
                                   "X1": float(right_center[0]),
                                   "Y1": float(right_center[1])})

            else:
                print(f"Eye Coordinates resulted Minus Coordinate")
                continue

        else:

            print(
                f"MediaPipe was failed to detect the faces on the image name {img_name}")
            continue
        
    print(image_count)
    savedDF = pd.DataFrame(data_array, columns=cols)
    savedDF.to_csv(csv_name)


if __name__ == "__main__":
    text_dir = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/labels/"

    text_files = np.array(glob(f"{text_dir}/*.txt"))
    mask = np.random.rand(len(text_files)) < 0.8
    train_text_files = text_files[mask]
    test_text_files = text_files[~mask]
    print(f"Number of Text Files:- {len(text_files)}")
    print(f"Number of Text files to TrainSet: {len(train_text_files)}")
    print(f"Number of Text files to TestSet: {len(test_text_files)}")
    
    

    img_dir = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/images/"
    saved_dir = "eyes"

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    trainAllCoords = mainAllCoords(train_text_files, img_dir,
                                   csvType="Train")

    mainCenterCsvCreation(trainAllCoords, img_dir,
                          saved_dir, "gi4eCentersTrain.csv")

    valAllCoords = mainAllCoords(test_text_files, img_dir,
                                 csvType="Test",)
    
    mainCenterCsvCreation(valAllCoords,img_dir,saved_dir,"gi4eCentersTest.csv")
