"""
    Pytorch implementation of following paper:
     https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9036930&isnumber=8833568
    
"""


import torch
import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision import models


def single_conv(in_ch, out_ch, dilation, pad, kernel_size):
    x = nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=pad,
            dilation=dilation,
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )

    return x


def double_conv(in_ch, out_ch, int_ch):
    x = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, int_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(int_ch),
        nn.ReLU(),
    )

    return x


def thribble_conv(in_ch, out_ch):
    x = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )

    return x


class ASPP(nn.Module):
    def __init__(self):
        super().__init__()

        self.avgPool = nn.AvgPool2d(kernel_size=32)
        self.up_1 = nn.Upsample(scale_factor=32)
        self.conv_1_0 = single_conv(512, 256, 1, 0, 1)

        self.conv_6 = single_conv(512, 256, 6, 6, 3)

        self.conv_12 = single_conv(512, 256, 12, 12, 3)

        self.conv_18 = single_conv(512, 256, 18, 18, 3)

        self.conv_1_1 = single_conv(512, 256, 1, 0, 1)

        self.conv_1_2 = single_conv(1280, 256, 1, 1, 3)

        self.conv_1_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.sig = nn.Sigmoid()

    def forward(self, x):

        avg_1 = self.avgPool(x)
        x1 = self.conv_1_0(avg_1)
        x1 = self.up_1(x1)  # [2, 256, 32, 32]

        d_6 = self.conv_6(x)  # [2, 256, 32, 32]
        d_12 = self.conv_12(x)  # [2, 256, 32, 32]
        d_18 = self.conv_18(x)  # [2, 256, 32, 32]
        d_1 = self.conv_1_1(x)  # [2, 256, 32, 32]

        x_2 = torch.cat([x1, d_6, d_12, d_18, d_1], axis=1)  # [2, 1280, 32, 32]
        x_2 = self.conv_1_2(x_2)  # [2, 256, 32, 32]

        x_2 = self.conv_1_3(x_2)  # [2, 512, 32, 32]

        x_2 = self.sig(x_2)  # [2, 512, 32, 32]

        # Element Wise Dot Produce
        x_3 = torch.mul(x_2, x)  # [2, 512, 32, 32]

        # Concatenation
        x_4 = torch.cat([x_3, x], axis=1)  # [2, 1024, 32, 32]

        return x_4


class IRISPARSENET(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_1 = double_conv(in_ch, 64, 64)
        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = double_conv(64, 128, 128)

        self.conv_3 = thribble_conv(128, 256)

        self.conv_4 = thribble_conv(256, 512)

        self.conv_5 = thribble_conv(512, 512)
        self.maxPool_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.aspp = ASPP()

        self.up_1 = nn.Upsample(scale_factor=2)

        self.conv_6 = double_conv(1024, 512, 256)

        self.up_2 = nn.Upsample(scale_factor=2)

        self.conv_7 = double_conv(768, 256, 128)

        self.up_3 = nn.Upsample(scale_factor=2)

        self.conv_8 = double_conv(384, 128, 64)

        self.up_4 = nn.Upsample(scale_factor=2)

        self.conv_9 = double_conv(192, 64, 32)

        self.conv_10 = single_conv(96, 32, 1, 1, 3)

        self.final = nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        s1 = self.conv_1(x)  # [2, 64, 512, 512]
        p1 = self.maxPool_1(s1)  # [2, 64, 256, 256]

        s2 = self.conv_2(p1)  # [2, 128, 256, 256]
        p2 = self.maxPool_1(s2)  # [2, 128, 128, 128]

        s3 = self.conv_3(p2)  # [2, 256, 128, 128]
        p3 = self.maxPool_1(s3)  # [2, 256, 64, 64]

        s4 = self.conv_4(p3)  # [2,512,64,64]
        p4 = self.maxPool_1(s4)  # [2,512,32,32]

        s5 = self.conv_5(p4)  # [2,512,32,32]
        p5 = self.maxPool_2(s5)  # [2,512,32,32]

        x_4 = self.aspp(p5)  # [2, 1024, 32, 32]

        x_4 = self.up_1(x_4)  # [2, 1024, 64, 64]

        x_5 = self.conv_6(x_4)  # [2, 256, 64, 64]

        x_5 = torch.cat([x_5, s4], axis=1)  # [2, 768, 64, 64]

        x_5 = self.up_2(x_5)  # [2, 768, 128, 128]

        x_6 = self.conv_7(x_5)  # [2,128,128,128]

        x_6 = torch.cat([x_6, s3], axis=1)  # [2, 384, 128, 128]

        x_6 = self.up_3(x_6)  # [2, 384, 256, 256]

        x_7 = self.conv_8(x_6)  # [2, 64, 256, 256]

        x_7 = torch.cat([x_7, s2], axis=1)  # [2, 192, 256, 256]

        x_7 = self.up_4(x_7)  # [2, 192, 512, 512]

        x_8 = self.conv_9(x_7)  # [2, 32, 512, 512]

        x_9 = torch.cat([x_8, s1], axis=1)  # [2, 96, 512, 512]

        x_10 = self.conv_10(x_9)  # [2, 32, 512, 512]

        final = self.final(x_10)  # [2, 2, 512, 512]
        
        return final


if __name__ == "__main__":

    tnsr = torch.randn([2, 3, 512, 512])
    tnsr = tnsr.to("cuda")
    model = IRISPARSENET(3, 2)
    model = model.to("cuda")
    print(model(tnsr).shape)
