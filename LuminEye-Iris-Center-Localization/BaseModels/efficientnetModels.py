import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import efficientnet_b3


class BB_model(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = efficientnet_b3(pretrained=True)

        layers = list(efficientnet.children())[:1]
        self.features1 = nn.Sequential(*layers)

        self.bb = nn.Sequential(
            nn.BatchNorm1d(1536),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 2),
        )

        # self.bb.apply(weights_init)

    def forward(self, x):
        x = self.features1(x)  # [1, 1536, 8, 8]
        x = F.relu(x)

        x = nn.AdaptiveAvgPool2d((1, 1))(x)  # [ 1,1536,1,1]

        x = x.view(x.shape[0], -1)  # [1,1536]

        return self.bb(x)


class CoordEfficientModel(nn.Module):
    def __init__(self, device, coordConv=True):
        super().__init__()

        if coordConv:
            self.conv1 = CoordConv2d(
                device, 3, 3, kernel_size=1, padding=0, stride=1, input_size=64
            )
        else:
            self.conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=0, stride=1)

        efficientnet = efficientnet_b3(pretrained=True)

        layers = list(efficientnet.children())[:1]
        self.features1 = nn.Sequential(*layers)

        self.bb = nn.Sequential(
            nn.BatchNorm1d(1536),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 2),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.features1(x)  # [1, 1536, 8, 8]
        x = F.relu(x)

        x = nn.AdaptiveAvgPool2d((1, 1))(x)  # [ 1,1536,1,1]

        x = x.view(x.shape[0], -1)  # [1,1536]

        return self.bb(x)


class CoordConv2d(nn.Module):

    def __init__(
        self,
        device,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        input_size,
    ):
        super(CoordConv2d, self).__init__()
        self.device = device
        self.cc_xy = self.make_channels(input_size)

        self.conv = nn.Conv2d(
            in_channels + 2,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def make_channels(self, input_size):
        coord_vals = (2 * torch.arange(input_size) / input_size) - 1

        xchannel = coord_vals.repeat((input_size, 1)).unsqueeze(dim=0)

        ychannel = xchannel.permute(0, 2, 1)

        return torch.cat((xchannel.unsqueeze(dim=0), ychannel.unsqueeze(dim=0)), dim=1)

    def forward(self, x):
        n = x.shape[0]
        x = torch.cat((x, self.cc_xy.repeat(n, 1, 1, 1).to(self.device)), dim=1)
        # print(x.shape)
        return self.conv(x)


if __name__ == "__main__":
    model = BB_model()
