import cv2
import albumentations as A
import numpy as np
from PIL import Image
import random
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches


image = Image.open("/home/nipun/Documents/Uni_Malta/Datasets/Datasets/UBRIS/train_img/C1_S1_I1.tiff")
mask = Image.open("/home/nipun/Documents/Uni_Malta/Datasets/Datasets/UBRIS/train_masks/C1_S1_I1.tiff")

def plot_examples(images):
    fig = plt.figure(figsize=(15,15))
    columns = 4
    rows = 5
    
    for i in range(1,len(images)):
        img = images[i-1]
        fig.add_subplot(rows,columns,i)
        plt.imshow(img)
    plt.show()



transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
    ]
)
images_list = [image]
image = np.array(image)
mask = np.array(mask)
for i in range(10):
    aug = transform(image=image,mask = mask)
    aug_image = aug["image"]
    aug_mask = aug["mask"]
    images_list.append(aug_image)
    images_list.append(aug_mask)
    
plot_examples(images_list)


