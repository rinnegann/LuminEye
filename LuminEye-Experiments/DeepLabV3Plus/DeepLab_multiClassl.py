import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from glob import glob
import segmentation_models_pytorch as smp
import time 
import wandb
import sys
sys.path.insert(1,"../utils/")
from losses import DiceLoss,IoU,pixel_wise_accuracy,get_lr

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_classes = 3
batch_size = 2

colors = [ [  0,   0,   0],[0,255,0],[0,0,255]]
label_colours = dict(zip(range(n_classes), colors))

valid_classes = [0,85, 170]
class_names = ["Background","Pupil","Iris"]


class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)


def decode_segmap(temp):
    #convert gray scale to color
    
    # print(temp.size())
    temp=temp.numpy()
    
    # temp = temp[:,:,0]
    # print(temp.shape())
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]
    
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

segmentation_classes = ["Background","Pupil","Iris"]


train_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Shorter_Dataset/train_img/"
train_masks  = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Shorter_Dataset/train_masks/"


val_images = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Shorter_Dataset/val_img"

val_masks =  "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/Miche/MICHE_MULTICLASS/Shorter_Dataset/val_masks/" 


train_x = sorted(
        glob(f"{train_images}/*"))
train_y = sorted(
        glob(f"{train_masks}/*"))
valid_x = sorted(
        glob(f"{val_images}/*"))
valid_y = sorted(
        glob(f"{val_masks }/*"))


def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l


def wandb_mask(bg_img, pred_mask, true_mask):
    
    return wandb.Image(bg_img,masks={"prediction" : {
          "mask_data" : pred_mask, 
          "class_labels" : labels()
      },"ground truth" : {
          "mask_data" : true_mask, 
          "class_labels" : labels()
      }})
    
    
def prediction_on_val(model,images,predictions,masks):
    
    mask_list = []
    
    count = 0 
    
    softmax= nn.Softmax(dim=1)
    unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    for x,y,z in zip(images,predictions,masks):
        
        for i in range(batch_size):
            
        
            # print(f"Image shape {x[i].size()}")
            
            # print(f"Masks shape {z[i].size()}")
            # print(f"Output shape {z[i].size()}")
        
            img = unorm(x[i]).permute(1,2,0).cpu().numpy()
        
            mask = z[i].cpu().numpy()
            
            pred = softmax(y[i]) # [3, 512, 512]
            
            # print(f"Prediction shape after Softmax: {pred.size()}")
            pred = torch.argmax(pred,dim=0).cpu().numpy()
            
    
            
            print(f"Image shape {img.shape}")
            print(f"Masks shape {mask.shape}")
            print(f"Output shape {pred.shape}")

        
        
            mask_list.append(wandb_mask(img,pred,mask))
        
    #     # if count ==15:
    #     #     break
        
     
        
        
    return mask_list




class Iris(Dataset):
    def __init__(self,images,masks,transform = None):
        self.transforms = transform
        self.images = images
        self.masks = masks
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))
        if self.transforms is not None:
            aug = self.transforms(image=img,mask=mask)
            img = aug['image']
            mask = aug['mask']
        return img,mask
    
def get_images(train_x,train_y,test_x,test_y,train_transform,val_transform,batch_size=1,shuffle=True,pin_memory=True):
    train_data = Iris(train_x,train_y,transform = train_transform)
    val_data  = Iris(test_x,test_y,transform =val_transform)
    train_batch = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_batch = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,drop_last=True)
    return train_batch,test_batch




train_transform = A.Compose([
    A.Resize(512,512),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                               rotate_limit=30, p=0.5),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                       b_shift_limit=25, p=0.5),
    A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.MotionBlur(p=0.3),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(512,512),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

train_batch,val_batch = get_images(train_x,train_y,valid_x,valid_y,train_transform,val_transform,batch_size=batch_size)

def visualize_multiclass_mask(train_batch):
    for img,mask in train_batch:
        print(img.shape)
        img1 = np.transpose(img[0,:,:,:],(1,2,0))
        mask1 = np.array(mask[0,:,:])
        img2 = np.transpose(img[1,:,:,:],(1,2,0))
        mask2 = np.array(mask[1,:,:])
        img3 = np.transpose(img[2,:,:,:],(1,2,0))
        mask3 = np.array(mask[2,:,:])
        fig , ax =  plt.subplots(3, 2, figsize=(18, 18))
        ax[0][0].imshow(img1)
        ax[0][1].imshow(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].imshow(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].imshow(mask3)
        break
    
x , y =next(iter(train_batch))

print(f' x = shape : {x.shape} ; type :{x.dtype}')
print(f' x = min : {x.min()} ; max : {x.max()}')
print(f' y = shape: {y.shape}; class : {y.unique()}; type: {y.dtype}')


def fit(epochs, model, train_loader, val_loader, optimizer, scheduler, patch=False):
    
    dice_loss = DiceLoss(mode="multiclass")
    train_losses = []
    test_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        #training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            #training phase
            image_tiles,mask_tiles = data
            # image_tiles= image_tiles.to(device)
            # mask_tiles = mask_tiles.to(device)
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1,c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)
            
            image = image_tiles.to(device); mask = mask_tiles.to(device);
            #forward
            output = model(image)
            loss =dice_loss(output,mask)
            #evaluation metrics
            iou_score += IoU(output, mask)
            accuracy += pixel_wise_accuracy(output, mask)
            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
            
            #step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step() 
            
            running_loss += loss.item()
            
        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            wb_images = []
            wb_prediction =[]
            wb_masks = []
            #validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    #reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data
                    image_titles = image_tiles.to(device)
                    mask_tiles = mask_tiles .to(device)
                    

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1,c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)
                    
                    image = image_tiles.to(device); mask = mask_tiles.to(device);
                    
                    wb_images.append(image)
                    wb_masks.append(mask)
                    output = model(image)
                    wb_prediction.append(output)
                    #evaluation metrics
                    val_iou_score +=  IoU(output, mask)
                    test_accuracy += pixel_wise_accuracy(output, mask)
                    #loss
                    loss = dice_loss(output,mask)                                  
                    test_loss += loss.item()
            
            #calculatio mean for each batch
            mask_list = prediction_on_val(model, wb_images, wb_prediction, wb_masks)
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))


            if min_loss > (test_loss/len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
                min_loss = (test_loss/len(val_loader))
                decrease += 1
                print('saving model...')
                torch.save(model, 'model-{:.3f}.pt'.format(val_iou_score/len(val_loader)))
                    

            
            #iou
            val_iou.append(val_iou_score/len(val_loader))
            train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy/ len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                  "Train IoU:{:.3f}..".format(iou_score/len(train_loader)),
                  "Val IoU: {:.3f}..".format(val_iou_score/len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))
            
            train_metrics = {"train/epoch":e+1,"train/train_loss":running_loss/len(train_loader),"train/iou":iou_score/len(train_loader),"train/accuracy":accuracy/len(train_loader)}
    
            val_metrics = {"train/epoch":e+1,"val/val_loss":test_loss/len(val_loader),"val/iou":val_iou_score/len(val_loader),"val/accuracy":test_accuracy/len(val_loader)}

            wandb.log({"predictions" : mask_list})
            wandb.log({**train_metrics, **val_metrics})
            
    
    
    wandb.finish()    
    
    history = {'train_loss' : train_losses, 'val_loss': test_losses,
               'train_miou' :train_iou, 'val_miou':val_iou,
               'train_acc' :train_acc, 'val_acc':val_acc,
               'lrs': lrs}
    
    
    
    
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history


if __name__ == "__main__":
    max_lr = 1e-3
    epoch = 10
    weight_decay = 1e-6
    
    model = smp.DeepLabV3Plus('efficientnet-b3',encoder_weights='imagenet', classes = n_classes, encoder_output_stride =16, activation=None,
                 encoder_depth= 5)
    model=model.to(device)
    experiment_name = "Short_Experiment"

    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_batch))



    config = {"epochs":epoch,
                    "max_learning_rate":max_lr,
                    "length_train":len(train_batch),
                    "length_val": len(val_batch),
                    "Augmentations":["ShiftScaleRotate","RGBShift","RandomBrightnessContrast","MotionBLur"],
                    "number_of_classes":n_classes,
                    "Resize_amt":(512,512),
                    "Base Model": "DeepLabV3Plus",
                    "BackBone":"EfficientNet-B3",
                    "Dataset": "Short",
                    "OPtimizer":"Adam",
                    "lr_scheduler": "OneCycleLR",
                    "Loss": "DiceLoss",
                    "weight_decay":weight_decay}


    wandb.init(project="LuminEys-Iris",entity="rinnegann",
            name=experiment_name,
            config=config)

    history = fit(epoch, model, train_batch, val_batch, optimizer, sched)