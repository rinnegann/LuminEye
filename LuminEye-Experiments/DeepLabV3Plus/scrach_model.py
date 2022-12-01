import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models.resnet import resnext50_32x4d
from torchvision.models.feature_extraction import get_graph_node_names
from pprint import pprint
from torchvision.models import resnext50_32x4d
from torchvision.models.feature_extraction import create_feature_extractor
import random
import seaborn as sns
import matplotlib.pyplot as plt

# model = resnext50_32x4d(weights="IMAGENET1K_V2")

return_nodes = ["layer1","layer3"]

# print(summary(model,(3,512,512)))

def conv(in_ch,out_ch,dilation=1,kernel_size=1):
  return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,padding="same",dilation=dilation,bias=False),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU())

class ASPP(nn.Module):

  def __init__(self,in_ch):

    super().__init__()
    
    self.avgpool = nn.AvgPool2d(kernel_size=32)
    self.conv_1 = conv(in_ch=in_ch,out_ch=256,dilation=1,kernel_size=1)
    self.up_1 = nn.Upsample(size=(32,32))

    self.conv_2 = conv(in_ch=in_ch,out_ch=256,dilation=1,kernel_size=1)

    self.conv_3 = conv(in_ch=in_ch,out_ch=256,dilation=6,kernel_size=1)

    self.conv_4 = conv(in_ch=in_ch,out_ch=256,dilation=12,kernel_size=1)

    self.conv_5 = conv(in_ch=in_ch,out_ch=256,dilation=18,kernel_size=1)

  def forward(self,x):
    conv_1 = self.avgpool(x)
    conv_1 = self.conv_1(conv_1)
    conv_1 = self.up_1(conv_1)

    conv_2 = self.conv_2(x)

    conv_3 = self.conv_3(x)

    conv_4 = self.conv_4(x)

    conv_5 = self.conv_5(x)

    return torch.cat([conv_1,conv_2,conv_3,conv_4,conv_5],axis=1)

class DeepLabv3Plus(nn.Module):
  def __init__(self,in_ch,out_ch=2):
      
      
      
        super().__init__()
        self.model =  resnext50_32x4d(weights="IMAGENET1K_V2")
        self.return_nodes = ["layer1","layer3"]
        self.feat_ext = create_feature_extractor(self.model,return_nodes=return_nodes)

        self.aspp = ASPP(in_ch)
        self.conv_1  = conv(1280,256)

        self.up_1 = nn.Upsample(scale_factor=4)
        
        self.conv_2 = conv(256,48)

        self.up_2 = nn.Upsample(scale_factor=4,mode="bilinear")

        self.last_conv = nn.Conv2d(304,out_ch,kernel_size=1)

  def forward(self,x):
      
        base_feature = self.feat_ext(x)
        x = self.aspp(base_feature[self.return_nodes[-1]])
        x = self.conv_1(x)
        x = self.up_1(x)

        x_1 = self.conv_2(base_feature[self.return_nodes[0]])

        x_2 = torch.cat([x,x_1],axis=1)

        x_2 = self.up_2(x_2)

        last = self.last_conv(x_2)

        return last
    
if __name__ == '__main__':
    in_tensor = torch.randn((2,3,512,512))
    deep_lab_model = DeepLabv3Plus(1024)
    print(deep_lab_model(in_tensor).shape)
