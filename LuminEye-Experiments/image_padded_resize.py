import cv2
import numpy as np
import os
from glob import glob


image_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Aspect_Ratio/train_image"
mask_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/utils/Images_with_Aspect_Ratio/train_masks"

saved_location = "./Image_Pad_MultiClass"

if not os.path.exists(saved_location):
    os.makedirs(saved_location)

images = sorted(glob(f"{image_path}/*"))
masks = sorted(glob(f"{mask_path}/*"))

desire_height = 128


def padded_resize(image_name, source, splt):

    print(image_name)
    color = [255, 255, 255]

    img = cv2.imread(image_name)

    img_name = image_name.split("/")[-1]

    height, width = img.shape[:2]

    print(f"Original Image width and heigh {width}|{height}")

    top = 0
    bottom = 0

    padded_img = None
    image_cropped = False
    

    if height > desire_height:
        print("Image height is greater than desired height")
        padded_img = img[0:desire_height, 0:width]
        image_cropped = True

    else:

        height_diff = desire_height - height

        if height_diff % 2 != 0:

            top = height_diff//2
            bottom = height_diff-top

        else:
            top = height_diff//2
            bottom = height_diff//2

        # padded_img= cv2.copyMakeBorder(img,top,bottom,0,0,cv2.BORDER_CONSTANT,value=color)

    if source == "image":

        # print(os.path.join(saved_location,f"{splt}_{source}"))
        if not os.path.exists(os.path.join(saved_location, f"{splt}_{source}")):
            os.makedirs(os.path.join(saved_location, f"{splt}_{source}"))

        if not image_cropped:
            print(f"Top {top} Bottom {bottom}")
            padded_img = cv2.copyMakeBorder(
                img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)

    else:
        # print(os.path.join(saved_location,f"{splt}_{source}"))
        if not os.path.exists(os.path.join(saved_location, f"{splt}_{source}")):
            os.makedirs(os.path.join(saved_location, f"{splt}_{source}"))

        if not image_cropped:

            padded_img = cv2.copyMakeBorder(
                img, top, bottom, 0, 0, cv2.BORDER_REPLICATE)

    cv2.imwrite(os.path.join(saved_location,
                f"{splt}_{source}", img_name), padded_img)


def main():

    count = 0
    for x, y in zip(images, masks):

        img = padded_resize(x, "image", "train")
        mask = padded_resize(y, "masks", "train")

        # if count == 0:
        #     break

    print("Images & Masks have been Resized")


if __name__ == "__main__":
    main()
