    import os
    import shutil
    from enum import Enum
    from typing import Any

    import torch
    from torch import nn
    from torch.nn import Module
    from torch.optim import Optimizer
    import math
    import random
    from typing import Any

    import cv2
    import numpy as np
    import torch
    from numpy import ndarray
    from torch import Tensor

    # from matplotlib import matplotlib as plt

    from utils import load_state_dict
    from model import srresnet_x4

    device = "cuda" if torch.cuda.is_available() else "cpu"



    img_path = "/home/nipun/Documents/Uni_Malta/Datasets/Datasets/UBRIS/train_img/"
    saved_location = "/home/nipun/Documents/Uni_Malta/Datasets/SuperResoultionDatasets/"
    dataset_name = "UBRIS"
    data_type = "train_img"
    model_type = "SRGAN"


    saved_location = os.path.join(saved_location,dataset_name,data_type,model_type)

    if not os.path.exists(saved_location):
        os.makedirs(saved_location)

    model_weights_path  = "results/pretrained_models/SRGAN_x4-ImageNet-8c4a7569.pth.tar"

    sr_model = srresnet_x4(in_channels=3,out_channels=3,channels=64,
                                                num_rcb=16)

    sr_model = sr_model.to(device=device)


    sr_model = load_state_dict(sr_model,model_weights_path)

    sr_model.eval()


    def preprocess_one_image(image_path: str, device: torch.device) -> Tensor:
        image = cv2.imread(image_path).astype(np.float32) / 255.0

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image data to pytorch format data
        tensor = image_to_tensor(image, False, False).unsqueeze_(0)

        # Transfer tensor channel image format data to CUDA device
        tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

        return tensor


    def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> Tensor:
        """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

        Args:
            image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
            range_norm (bool): Scale [0, 1] data to between [-1, 1]
            half (bool): Whether to convert torch.float32 similarly to torch.half type

        Returns:
            tensor (Tensor): Data types supported by PyTorch

        Examples:
            >>> example_image = cv2.imread("lr_image.bmp")
            >>> example_tensor = image_to_tensor(example_image, range_norm=True, half=False)

        """
        # Convert image data type to Tensor data type
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

        # Scale the image data from [0, 1] to [-1, 1]
        if range_norm:
            tensor = tensor.mul(2.0).sub(1.0)

        # Convert torch.float32 image data type to torch.half image data type
        if half:
            tensor = tensor.half()

        return tensor


    for image in os.scandir(img_path):
        
        image_name = image.path.split("/")[-1].split(".")[0]+".png"
        
        
        if image.is_file():
            lr_tensor = preprocess_one_image(image.path, device)
            
        with torch.no_grad():
            sr_tensor = sr_model(lr_tensor)
            
        predicted_image = sr_tensor.squeeze(0).permute(1,2,0).mul(255).clamp(0,255).cpu().numpy().astype("uint8")
        sr_image = cv2.cvtColor(predicted_image, cv2.COLOR_RGB2BGR)
        print(sr_image.shape)
        cv2.imwrite(os.path.join(saved_location,image_name),sr_image)
        torch.cuda.empty_cache()

