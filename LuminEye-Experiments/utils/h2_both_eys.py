import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe
import pandas as pd


def cropped_both_eyes(img, shape_array, padded_amt=15):
    """Cropped eye region 

    Args:
        img (__numpy__): _Original Image_
        shape_array (_numpy_): _FaceLandMark locations_
        padded_amt (int, optional): _padding size_. Defaults to 15.

    """

    FaceLandmaks = {
        "top_left": shape_array[70], "bottom_right": shape_array[345]}

    return img[FaceLandmaks["top_left"][1]:FaceLandmaks["bottom_right"][1],
               FaceLandmaks["top_left"][0]:FaceLandmaks["bottom_right"][0]], FaceLandmaks
    
    
    
    
def mpArrayToNumpy(landmark_array, img):

    shape_arr = []

    for landmark in landmark_array.landmark:
        x = landmark.x
        y = landmark.y

        relative_x = int(x * img.shape[1])
        relative_y = int(y * img.shape[0])

        shape_arr.append([relative_x, relative_y])

    return np.array(shape_arr)


mp_face_mesh = mediapipe.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


def substract_list(l1, l2):

    return [abs(l1[0]-l2[0]), abs(l1[1]-l2[1])]


saved_dir = "./H2DATASET_BOTH_EYES"


if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)





IMG_PATH = '/home/nipun/Documents/Uni_Malta/Datasets/I2HEAD'

df = pd.read_csv(
    '/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Iris-Center-Localization/i2head_annotations.csv')


image_count = 0
visualize = True
data_array = []



def rescaleAccordingToImageCrop(bbox, Eyes):
    
    
    left_center = bbox[2:4]
    right_center = bbox[8:10]

    left_center[0] = float(left_center[0]) - Eyes['top_left'][0]
    left_center[1] = float(left_center[1]) - Eyes['top_left'][1]

    right_center[0] = float(right_center[0]) - Eyes['top_left'][0]
    right_center[1] = float(right_center[1]) - Eyes['top_left'][1]

    return left_center, right_center




cols = ["Image_Name", 'L_X1', 'L_Y1','R_X1', 'R_Y1']
for idx, row in df.iterrows():

    img_path = os.path.join(IMG_PATH, row["ImageName"].replace('"', '')[7:])

    print(img_path)
    bbox = list(map(float, row["Coordinates"][1:-1].split(",")))

    
    if not os.path.exists(img_path):
        continue
    
    img = cv2.imread(img_path)

    results = face_mesh.process(img)
    
    print(results.multi_face_landmarks)
    
    
    

    if results.multi_face_landmarks is not None:
        image_count += 1

        img_name = f'I2head_{image_count}'

        landmarks = results.multi_face_landmarks[0]

        shape_arr = mpArrayToNumpy(landmarks, img)

        BothEyes, FaceLandmaks = cropped_both_eyes(
            img, shape_arr)

        left_center, right_center = rescaleAccordingToImageCrop(
            bbox, FaceLandmaks)

        if BothEyes.shape[0] > 20:

            img_eyes = BothEyes.copy()

            # cv2.circle(BothEyes, (int(left_center[0]), int(
            #     left_center[1])), 1, (0, 0, 255), -1)

            # cv2.circle(BothEyes, (int(right_center[0]), int(
            #     right_center[1])), 1, (0, 0, 255), -1)

            # fig, axs = plt.subplots(1)

            # axs.set_title("Both Eyes")
            # # axs[1].set_title("Right Eye")

            # axs.axis("off")

            # axs.imshow(BothEyes[:, :, ::-1])

            # plt.tight_layout()
            # plt.show()
            # plt.close('all')

            # # Normlaize the Center Coordinates before send to the CSV file
            data_array.append({"Image_Name": f"{img_name}.png",
                               "L_X1": float(left_center[0])/BothEyes.shape[1],
                               "L_Y1": float(left_center[1])/BothEyes.shape[0],
                               "R_X1": float(right_center[0])/BothEyes.shape[1],
                               "R_Y1": float(right_center[1])/BothEyes.shape[0]})

            cv2.imwrite(os.path.join(
                saved_dir, f"{img_name}.png"), img_eyes)
        else:
            continue

    else:
        print(
            f"MediaPipe was failed to detect the faces on the image name {img_name}")
        continue
    
    
df = pd.DataFrame(data_array, columns=cols)

df.to_csv("H2head_BothEyes.csv")
