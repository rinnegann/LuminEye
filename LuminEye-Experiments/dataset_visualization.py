"""Copyright(C) Univercity Malta-All Rights Reserved Unauthorized copying of this file,via any medium is stricly prohibited
   Proprietary and confidential Written by Nipun Sandamal<nipunsandamal1997@gmal.com>"""
   
import cv2
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt
import numpy as np


class DatasetVisualizer:
    def __init__(self, dataset_path: str, dataset_name: str):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        self.images = "images"
        self.labels = "labels"

    def visualize_datasets(self, annot_ext: str):
        #self.bbox_gi4e(annot_ext)
        self.bbox_bioid(annot_ext)

    def bbox_bioid(self, annot_ext):

        images_sample = random.sample(sorted(glob.glob(os.path.join(
            self.dataset_path, self.images) + "/**", recursive=True)),3)

        fig, ax = plt.subplots(3, 1, figsize=(50, 50))

        ax = ax.ravel()
        for idx, img_f in enumerate(images_sample):
            image = cv2.imread(img_f)

            txt_file = os.path.join(self.dataset_path, self.labels, img_f.split(
                "/")[-1].split(".")[0] + "." + annot_ext)

            with open(txt_file, "r") as f:
                data = [int(x)
                        for x in f.readlines()[1].rstrip("\n").split("\t")]

            cv2.circle(image, (data[0], data[1]),
                       radius=0, color=(0, 0, 255), thickness=2)
            cv2.circle(image, (data[2], data[3]),
                       radius=0, color=(0, 0, 255), thickness=3)

            ax[idx].imshow(image, cmap="gray")
            ax[idx].set_title(img_f.split("/")[-1].split(".")[0])

        plt.show()

    def bbox_gi4e(self, annot_ext):

        text_file_sample = random.sample(sorted(glob.glob(os.path.join(
            self.dataset_path, self.labels) + "/**", recursive=True)), 1)

        fig, ax = plt.subplots(3, 4, figsize=(25, 25))

        ax = ax.ravel()

        with open(text_file_sample[0], "r") as f:
            data = [s.rstrip("\n").split("\t") for s in f.readlines()]

        for idx, image_data in enumerate(data):

            image = cv2.imread(os.path.join(
                self.dataset_path, self.images, image_data[0]))

            for coord in list(self.divide_chunks(image_data[1:], 2)):

                cv2.circle(image, (int(float(coord[0])), int(
                    float(coord[1]))), radius=0, color=self.random_color(), thickness=3)

            ax[idx].imshow(image, cmap="gray")
            ax[idx].set_title(image_data[0])
        plt.show()

    def random_color(self):
        rgbl = [255, 0, 0]
        random.shuffle(rgbl)
        return tuple(rgbl)

    def divide_chunks(self, l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]


if __name__ == "__main__":

    DataSetName = "BioId"
    DataSetPath = "/home/nipun/Documents/Uni_Malta/Datasets/BioID-FaceDatabase-V1.2/"

    DV = DatasetVisualizer(DataSetPath, DataSetName)

    DV.visualize_datasets("eye")
    # DataSetName = "gi4e"
    # DataSetPath = "/home/nipun/Documents/Uni_Malta/Datasets/gi4e_database/"

    # DV = DatasetVisualizer( DataSetPath, DataSetName)

    # DV.visualize_datasets("txt")
