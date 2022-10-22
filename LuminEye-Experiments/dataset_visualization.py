
import cv2
import numpy as np
import os
import random 
import glob
import matplotlib.pyplot as plt
import numpy as np


class DatasetVisualizer:
    def __init__(self,dataset_path:str,dataset_name:str):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        self.images = "images"
        self.labels = "labels"

    def visualize_datasets(self,annot_ext:str):
        images_sample = random.sample(sorted(glob.glob(os.path.join(self.dataset_path,self.images) + "/**", recursive=True)),15)

        fig,ax = plt.subplots(3,5,figsize=(15,15))
        
        ax = ax.ravel()
        for idx,img_f in enumerate(images_sample):
            
            print()
            image = cv2.imread(img_f)
            txt_file = os.path.join(self.dataset_path,self.labels,img_f.split("/")[-1].split(".")[0] +"."+ annot_ext)

            with open(txt_file,"r") as f:
                data = [int(x) for x in f.readlines()[1].rstrip("\n").split("\t")]
                
                
            cv2.circle(image, (data[0],data[1]), radius=0, color=(0, 0, 255), thickness=2)
            cv2.circle(image, (data[2],data[3]), radius=0, color=(0, 0, 255), thickness=3)

            
            ax[idx].imshow(image,cmap="gray")
            ax[idx].set_title(img_f.split("/")[-1].split(".")[0])
            
        plt.show()

        
        
        
        



if __name__ == "__main__":

    DataSetName = "BioId"
    DataSetPath = "/home/nipun/Documents/Uni_Malta/Datasets/BioID-FaceDatabase-V1.2/"

    DV = DatasetVisualizer( DataSetPath, DataSetName)

    DV.visualize_datasets("eye")
