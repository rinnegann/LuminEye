import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe
import pandas as pd


def mpArrayToNumpy(landmark_array, img):

    shape_arr = []

    for landmark in landmark_array.landmark:
        x = landmark.x
        y = landmark.y

        relative_x = int(x * img.shape[1])
        relative_y = int(y * img.shape[0])

        shape_arr.append([relative_x, relative_y])

    return np.array(shape_arr)


def cropped_image(img, shape_array, padded_amt=30):
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
                   15
                   , Leye["top_left"][0]:Leye["bottom_right"][0]]

    right_eye = img[Reye["top_left"][1]:Reye["bottom_right"][1] +
                    30, Reye["top_left"][0]-5:Reye["bottom_right"][0]]
    
    
    
    Reye['top_left'][0] = Reye['top_left'][0] - 5

    return left_eye, right_eye, Leye, Reye

mp_face_mesh = mediapipe.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


saved_dir = "./eyes"


if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)
    

IMG_PATH = '/home/nipun/Documents/Uni_Malta/Datasets/I2HEAD'

df = pd.read_csv('/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Iris-Center-Localization/i2head_annotations.csv')


image_count = 0
visualize = True
data_array =[]
for idx,row in df.iterrows():
    
    img_path = os.path.join(IMG_PATH,row["ImageName"].replace('"','')[7:])
    
    
    
    print(img_path)
    
    print(row["ImageName"].replace('"','')[7:])
    
    bbox = row["Coordinates"][1:-1].split(",")
    
    left_center = bbox[2:4]
    
    right_center = bbox[8:10]
    
    img = cv2.imread(img_path)

    
    
    results = face_mesh.process(img)
    
    if results.multi_face_landmarks is not None:
        image_count +=1
        
        img_name = f'I2head_{image_count}'
        
        landmarks = results.multi_face_landmarks[0]

        shape_arr = mpArrayToNumpy(landmarks, img)

        left_eye, right_eye, Leye, Reye = cropped_image(
            img, shape_arr)
        
      
        left_center[0] = float(left_center[0]) - Leye['top_left'][0]
        left_center[1] = float(left_center[1]) - Leye['top_left'][1]
    
        
        right_center[0] = float(right_center[0]) -  Reye['top_left'][0]
        right_center[1] = float(right_center[1]) -  Reye['top_left'][1]
            
 


        cv2.imwrite(os.path.join(
            saved_dir, f"{img_name}_left.png"), left_eye)

        cv2.imwrite(os.path.join(
            saved_dir, f"{img_name}_right.png"), right_eye)

        if visualize:

            cv2.circle(left_eye, (int(left_center[0]), int(
                left_center[1])), 1, (0, 0, 255), -1)
            cv2.circle(right_eye, (int(right_center[0]), int(
                right_center[1] )), 1, (0, 0, 255), -1)

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
        data_array.append({"Image_Name": f"{img_name}_left.png",
                            "X1": float(left_center[0])/left_eye.shape[1],
                            "Y1": float(left_center[1])/left_eye.shape[0]})

        data_array.append({"Image_Name": f"{img_name}_right.png",
                            "X1": float(right_center[0])/right_eye.shape[1],
                            "Y1": float(right_center[1])/right_eye.shape[0]})
    
    else:
        print(
            f"MediaPipe was failed to detect the faces on the image name {img_name}")
        continue

    # break