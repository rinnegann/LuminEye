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


def getCenterofEye(coord, leye, reye):
    

    left_center = coord[8:10]
    right_center = coord[2:4]

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
            "bottom_right": shape_array[446]}

    left_eye = img[Leye["top_left"][1]:Leye["bottom_right"][1] +
                   padded_amt, Leye["top_left"][0]:Leye["bottom_right"][0]]

    right_eye = img[Reye["top_left"][1]:Reye["bottom_right"][1] +
                    padded_amt, Reye["top_left"][0]:Reye["bottom_right"][0]]

    return left_eye, right_eye, Leye, Reye


def main(text_dir,img_dir,saved_dir,visualize = True):
    data_array = []
    cols = ["Image_Name", 'X1', 'Y1']

    mp_face_mesh = mediapipe.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    
    for text_path in os.scandir(text_dir):

        if text_path.is_file():

            with open(text_path.path, "r") as f:
                info = [k.rstrip().split("\t") for k in f.readlines()]

            for obj in info:

                image_filepath = os.path.join(img_dir, obj[0])

                print(image_filepath)

                img = cv2.imread(image_filepath)

                results = face_mesh.process(img)

                if results.multi_face_landmarks is not None:
                    landmarks = results.multi_face_landmarks[0]

                    shape_arr = mpArrayToNumpy(landmarks, img)

                    left_eye, right_eye, Leye, Reye = cropped_image(
                        img, shape_arr)

                    left_center, right_center = getCenterofEye(
                        coord=obj[1:], leye=Leye, reye=Reye)

                    cv2.imwrite(os.path.join(
                        saved_dir, f"{obj[0].split('.')[0]}_left.png"), left_eye)

                    cv2.imwrite(os.path.join(
                        saved_dir, f"{obj[0].split('.')[0]}_right.png"), right_eye)

                    if visualize:

                        cv2.circle(left_eye, (int(left_center[0]), int(
                            left_center[1])), 1, (0, 0, 255), -1)
                        cv2.circle(right_eye, (int(right_center[0]), int(
                            right_center[1])), 1, (0, 0, 255), -1)

                        fig, axs = plt.subplots(1, 2)

                        axs[0].set_title("Left Eye")
                        axs[1].set_title("Right Eye")

                        axs[0].axis("off")
                        axs[1].axis("off")

                        axs[0].imshow(left_eye[:, :, ::-1])
                        axs[1].imshow(right_eye[:, :, ::-1])

                        plt.tight_layout()
                        plt.show()
                        plt.close('all')

                    # Normlaize the Center Coordinates before send to the CSV file
                    data_array.append({"Image_Name": f"{obj[0].split('.')[0]}_left.png",
                                       "X1": left_center[0]/left_eye.shape[1],
                                       "Y1": left_center[1]/left_eye.shape[0]})

                    data_array.append({"Image_Name": f"{obj[0].split('.')[0]}_right.png",
                                       "X1": right_center[0]/right_eye.shape[1],
                                       "Y1": right_center[1]/right_eye.shape[0]})

                else:
                    print(
                        f"MediaPipe was failed to detect the faces on the image name {obj[0]}")
                    continue
    
    df = pd.DataFrame(data_array, columns=cols)
    
    
    
    df.to_csv("gi4e_center.csv")

if __name__ == "__main__":
    text_dir = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/labels/"
    img_dir = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/images/"
    saved_dir = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/eyes"

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    main(text_dir,img_dir,saved_dir,visualize=False)