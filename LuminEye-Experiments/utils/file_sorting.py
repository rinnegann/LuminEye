"""This python code is responsible to generate data in proper format of H2HEAD Dataset"""
import os
import shutil
import pandas as pd


annot_path = '/home/nipun/Documents/Uni_Malta/Datasets/MPIIGaze_subset_annotations/filenames.json'

coord_annot = "/home/nipun/Documents/Uni_Malta/Datasets/MPIIGaze_subset_annotations/annotations.json"

saved_location = "/home/nipun/Documents/Uni_Malta/Datasets/MP2GAZE"


if not os.path.exists(saved_location):
    os.makedirs(saved_location)


def main(file_name,bbox_name):

    with open(annot_path,'r') as f:
        data = [line.rstrip() for line in f.readlines()][1:-1]
        
        
    file_names = []
    for f_path in data:
        
        f_path = f_path[:-1].replace(" ",'').split('\\')
        
        
        f_path = '/'.join(list(filter(lambda x: len(x)>1,f_path)))

        file_names.append(f_path[1:])
        
    with open(bbox_name,'r') as r:
        bbox = r.read().split(']')
        
    
    coords = []
    for bb in bbox:

        bb_array = bb.rstrip().lstrip(',').strip().replace("[",' ')
        
        if len(bb_array.split(',')) >1 :

    #             # print(bb_array.split(','))
      
            coords.append([float(x.lstrip()) for x in bb_array.split(',')])
    df = pd.DataFrame(columns=['ImageName','Coordinates'])
    
    
    
    assert  len(coords) == len(file_names)
    
    
    for x,y in zip(file_names,coords):
        df = df.append({'ImageName':x,'Coordinates':y},ignore_index=True)
        
    df.to_csv('mp2gaze_annotations.csv')
   

if __name__ == '__main__':
    main(annot_path,coord_annot)
