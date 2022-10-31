import torch
import torch.nn as nn

class double_conv(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(in_c,out_c,kernel_size=3,padding=1)
        self.bn_1 = nn.BatchNorm2d(out_c)
        
        self.conv_2 = nn.Conv2d(out_c,out_c,kernel_size=3,padding=1)
        self.bn_2 = nn.BatchNorm2d(out_c)
        
        self.relu = nn.ReLU()
        
    def forward(self,inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.relu(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        
        return x
    
class Encoder_Block(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv = double_conv(in_c,out_c)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self,inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        
        return x,p
    
class Decoder_Block(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_c,out_c,kernel_size=2,stride=2,padding=0)
        self.conv = double_conv(out_c +out_c,out_c)
        
    def forward(self,inputs,skip):
        x = self.up(inputs)
        #print(f"Target Tesnor Size {x.size()}| Skip Tensor:{skip.size()}")
        x = torch.cat([x,skip],axis=1)
        x = self.conv(x)

        return x
    
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder_Block
        self.e1 = Encoder_Block(3, 64)
        self.e2 = Encoder_Block(64, 128)
        self.e3 = Encoder_Block(128,256)
        self.e4 = Encoder_Block(256,512)
        
        # Bridge 
        self.b = double_conv(512, 1024)
        
        # Decoder_Block
        self.d1 = Decoder_Block(1024,512)
        self.d2 = Decoder_Block(512,256)
        self.d3 = Decoder_Block(256,128)
        self.d4 = Decoder_Block(128,64)
        
        # Classification
        self.out  = nn.Conv2d(64,1,kernel_size=1,padding=0)
        
    def forward(self,inputs):
        
        # Encoder
        s1,p1 = self.e1(inputs)
        s2,p2 = self.e2(p1)
        s3,p3 = self.e3(p2)
        s4,p4 = self.e4(p3)
        
        # Bridge
        b = self.b(p4)
        
        # Decoder
        d1 = self.d1(b,s4)
        d2 = self.d2(d1,s3)
        d3 = self.d3(d2,s2)
        d4 = self.d4(d3,s1)
        
        outputs = self.out(d4)
        
        return outputs
        
    

if __name__ == '__main__':
    x = torch.rand([2,3,512,512])
    model = UNET()
    pred = model(x)
    
    print(f"Tensor Shape is {pred.size()}")
 
 
        