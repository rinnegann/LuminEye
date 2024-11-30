import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class BB_model(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet34(weights=True)

        # for param in resnet.parameters():

        #     param.requires_grad = False

        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])

        self.bb = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features1(x)  # 1, 128, 32, 32

        x = self.features2(x)  # [1, 512, 8, 8]

        x = F.relu(x)

        x = nn.AdaptiveAvgPool2d((1, 1))(x)  # [ 1,512,1,1]

        x = x.view(x.shape[0], -1)  # [1, 512]

        return self.bb(x)
