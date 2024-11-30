import os
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from PIL import Image, ImageDraw
from utils import *
from dataset_classes import CenterDatasetRG


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


# regression_model_path = '/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Iris-Center-Localization/BEST_RESNET_REGRESSION_MODEL_FOR_CROPPED_EYES/Regression_model_1.487574208665777.pth'

regression_model_path = "/home/nipun/Documents/Uni_Malta/LuminEye/LuminEye-Iris-Center-Localization/RecentModels/SpheroPipeLine/Resnet_32_IMG_SIZE__32/Regression_model_0.25639680131442016.pth"


IMAGE_DIR = (
    "/home/nipun/Documents/Uni_Malta/Datasets/CenterRegression/MixDataset/images"
)

val_df = pd.read_csv(
    "/home/nipun/Documents/Uni_Malta/Datasets/CenterRegression/MixDataset/valAll.csv"
)


RESIZE_AMT = 32
BACTH_SIZE = 8


val_transforms = A.Compose(
    [
        A.Resize(width=RESIZE_AMT, height=RESIZE_AMT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1),
    ]
)


test_ds = CenterDatasetRG(val_df, transforms=val_transforms)


testLoader = DataLoader(
    test_ds,
    batch_size=BACTH_SIZE,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=True,
)


def main(model, test_loader, save_location):

    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    image_count = 0

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    for x, y, img_id in test_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            out_coord = model(x)

        for i in range(x.shape[0]):

            image_name = img_id[i]

            image = transforms.ToPILImage()(unnorm(x[i]))
            gt_coord = y[i].detach().cpu().numpy()
            pred_coord = out_coord[i].detach().cpu().numpy()

            draw = ImageDraw.Draw(image)

            # gt --> red
            draw.point((int(gt_coord[0]), int(gt_coord[1])), "red")

            # prediction --> green
            draw.point((int(pred_coord[0]), int(pred_coord[1])), "green")

            image.save(os.path.join(saved_location, image_name))


if __name__ == "__main__":

    saved_location = "/home/nipun/Documents/Uni_Malta/Datasets/CenterRegression/MixDataset/RecentModelForgi4ebioidI2HEAD"
    regression_model = torch.load(regression_model_path, map_location=device)
    main(regression_model, testLoader, saved_location)
