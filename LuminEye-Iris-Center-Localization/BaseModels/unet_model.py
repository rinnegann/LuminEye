import torch
import torch.nn as nn


def double_conv(input_channels,output_channels):
    return nn.Sequential(nn.Conv2d(input_channels,output_channels,kernel_size=3,stride=1,padding='same'),
                         nn.BatchNorm2d(output_channels),
                         nn.ReLU(),
                         nn.Conv2d(output_channels,output_channels,kernel_size=3,stride=1,padding='same'),
                         nn.BatchNorm2d(output_channels),
                         nn.ReLU())
    
class UNET(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        
        self.num_classes = num_classes
        self.conv_1 = double_conv(3,64)
        self.mx_1 = nn.MaxPool2d(kernel_size=(2,2))
        
        
        self.conv_2 = double_conv(64,128)
        self.mx_2 = nn.MaxPool2d(kernel_size=(2,2))
        
        
        self.conv_3 = double_conv(128,256)
        self.mx_3 = nn.MaxPool2d(kernel_size=(2,2))
        
        
        
        self.conv_4 = double_conv(256,512)
        self.mx_4 = nn.MaxPool2d(kernel_size=(2,2))
        
        
        
        self.conv_5 = double_conv(512,1024)
        
        
        
        self.up_1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        
        
        
        self.conv6 = double_conv(1024,512)
       
       
        self.up_2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        
        
        
        self.conv7 = double_conv(512,256)
        
        
        self.up_3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        
        
        self.conv8 = double_conv(256,128)
        

        self.up_4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        
        self.conv9 = double_conv(128,64)
        
        self.out_conv = nn.Conv2d(64,self.num_classes,kernel_size=1)
        
    def forward(self,x):

        
        x1 = self.conv_1(x)  # 1, 64, 64,64
        
        x1_max = self.mx_1(x1)
        
        
        
        x2 = self.conv_2(x1_max)  # 1, 128, 32, 32
        x2_max = self.mx_2(x2)
        
        x3 = self.conv_3(x2_max)  # 1, 256, 16,16
        x3_max = self.mx_3(x3)
        
        
        x4 = self.conv_4(x3_max) # 1, 512, 8, 8
        x4_max = self.mx_4(x4)
        
        
        
        # Bridge
        
        x5 = self.conv_5(x4_max)  # 1, 1024, 4, 4
       
        
        
        x6 = self.up_1(x5) # [1, 512, 8, 8]
      
        
        
        decoder_1 = torch.concat([x6,x4],dim=1) # 1, 1024, 8, 8]
        
        
        decoder_1 = self.conv6(decoder_1)  # [1, 512, 8, 8]
        
        
        
        
        d_2 = self.up_2(decoder_1)
        
        decoder_2 = torch.concat([d_2,x3],dim=1) # [1, 512, 16, 16]
        
        decoder_2 = self.conv7(decoder_2) # [1, 256, 16, 16]
        
       
        d_3 = self.up_3(decoder_2)
        
        decoder_3 = torch.concat([d_3,x2],dim=1)
        
        decoder_3 = self.conv8(decoder_3) # [1, 128, 32, 32]
        
        
        d_4 = self.up_4(decoder_3)
        decoder_4 = torch.concat([d_4,x1],dim=1)
        
        decoder_4 = self.conv9(decoder_4) # 1, 64, 64, 64
        
        output = self.out_conv(decoder_4).view(-1,self.num_classes*x.shape[2]*x.shape[3])
        
        return output