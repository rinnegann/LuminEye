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


def cropped_image(img, shape_array, padded_amt=15):
    """Cropped eye region 

    Args:
        img (__numpy__): _Original Image_
        shape_array (_numpy_): _FaceLandMark locations_
        padded_amt (int, optional): _padding size_. Defaults to 15.

    """

    Leye = {"top_left": shape_array[70], "bottom_right": shape_array[133]}

    Reye = {"top_left": shape_array[300],
            "bottom_right": shape_array[362]}

    left_eye = img[Leye["top_left"][1]:Leye["bottom_right"][1] +
                    padded_amt
                   , Leye["top_left"][0]:Leye["bottom_right"][0]]

    right_eye = img[Reye["top_left"][1]:Reye["bottom_right"][1] +
                    padded_amt, Reye["top_left"][0]:Reye["bottom_right"][0]]
    
    
    
    print(Reye)
    
    Reye['top_left'][0] = Reye['top_left'][0] - 5

    return left_eye, right_eye, Leye, Reye

mp_face_mesh = mediapipe.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def substract_list(l1,l2):  
    
    return [abs(l1[0]-l2[0]), abs(l1[1]-l2[1])]


saved_dir = "./checkDataset"


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
    
    bbox = list(map(float,row["Coordinates"][1:-1].split(",")))
    
    print(bbox)
    
    left_center = bbox[2:4]
    
    left_inner_center = bbox[0:2]
    
    
    

    left_outer_center  = bbox[4:6]
    
    
    print("#" * 10)
    print("Left Eye")
    print(f"Differece Between Inner EYE & Center {substract_list(left_center,left_inner_center)}")
    print(f"Differece Between outer EYE & Center {substract_list(left_center,left_outer_center)}")
    
    
    
    right_center = bbox[8:10]
    
    
    right_inner_center = bbox[6:8]
    
    right_outer_center = bbox[10:12]
    print("*" * 10)
    print("Right Eye")
    print(f"Differece Between Inner EYE & Center {substract_list(right_center,right_inner_center) }")
    print(f"Differece Between outer EYE & Center {substract_list(right_center,right_outer_center)}")
    
    print("#" * 10)
    
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