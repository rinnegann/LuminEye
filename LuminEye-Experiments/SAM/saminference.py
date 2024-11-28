
import torch
import os
import numpy as np
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2
from segment_anything import sam_model_registry,SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide




device = "cuda" if torch.cuda.is_available() else "cpu"


sam_checkpoint ="sam_vit_h_4b8939.pth"
model_type = "vit_h"


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to("cpu")
torch.cuda.empty_cache()

img_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Experiments/YoloV8/EyeDataset/val/images/001_IP5_IN_F_RI_01_2.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


yolo_model_path = "/home/nipun/Music/yolov5/runs/train/exp8/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)
model.to(device)


results = model(img,size=640) 


iris_bbox = results.xyxy[0][0].detach().cpu().numpy()[:4]
pupil_box = results.xyxy[0][1].detach().cpu().numpy()[:4]


image_bbox = torch.tensor([iris_bbox, pupil_box],device=sam.device)



predictor = SamPredictor(sam)
predictor.set_image(img)


masks,_,_ = predictor.predict(
    point_coords = None,
    point_labels = None,
    box = iris_bbox[None,:],
    multimask_output =False
)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
    
    
plt.figure(figsize=(10, 10))
plt.imshow(img)
show_mask(masks[0], plt.gca())
show_box(iris_bbox, plt.gca())
plt.axis('off')
plt.show()


