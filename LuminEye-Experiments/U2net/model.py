import torch
import torch.nn as nn




class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, rate=1):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels,kernel_size=3,padding="same",dilation=rate)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act(x)
        return x

class RSU_L(nn.Module):
  def __init__(self,in_ch,out_ch,int_ch,num_layers,rate=2):
    super().__init__()

    self.num_layers = num_layers
    self.skip = []
    self.int_ch = int_ch
    self.conv_1 = conv_block(in_ch,out_ch)

    self.conv_2 = conv_block(out_ch,int_ch)

    self.maxPool = nn.MaxPool2d(kernel_size=2)

    self.conv_3 = conv_block(int_ch,int_ch)


    # Bridge
    self.conv_4 = conv_block(int_ch,int_ch,rate=2)

    self.conv_5 = conv_block(int_ch+int_ch,int_ch)

    self.up = nn.Upsample(scale_factor=2,mode="bilinear")

    self.conv_6 = conv_block(int_ch+int_ch,out_ch)

  def forward(self,x):
    x = self.conv_1(x)
    init_feat = x

    x = self.conv_2(x)
    self.skip.append(x)

    for _ in range(self.num_layers-2):
      x = self.maxPool(x)
      x = self.conv_3(x)
      self.skip.append(x)

    self.skip.reverse()   # torch.Size([2, 32, 16, 16])
    # [print(j.shape) for j in self.skip]
    x = self.conv_4(x) #[2, 32, 16, 16]


    x = torch.cat([x,self.skip[0]],axis=1) # [2, 64, 16, 16]

    x = self.conv_5(x) # [2, 32, 16, 16]

    for i in range(self.num_layers-3):

        x = self.up(x)
        x = torch.cat([x,self.skip[i+1]],axis=1)

        x = conv_block(x.shape[1],self.int_ch)(x)


    x = self.up(x)  # [2, 32, 512, 512]
    x = torch.cat([x,self.skip[-1]],axis=1) # [2, 64, 512, 512]


    x = self.conv_6(x)  # [2, 64, 512, 512]

    # print(init_feat.shape) [2, 64, 512, 512]

    x = x + init_feat # [2, 64, 512, 512]

    return x

class RSU_4F(nn.Module):
  def __init__(self,in_ch,out_ch,int_ch):

    super().__init__()

    # Initial Conv
    self.conv_1 = conv_block(in_ch,out_ch,rate=1)


    #Encoder Part
    self.conv_2 = conv_block(out_ch,int_ch,rate=1)
    self.conv_3 = conv_block(int_ch,int_ch,rate=2)
    self.conv_4 = conv_block(int_ch,int_ch,rate=4)

    # Bridge
    self.conv_5 = conv_block(int_ch,int_ch,rate=8)

    # Decoder
    self.conv_6 = conv_block(int_ch+int_ch,int_ch,rate=4)

    self.conv_7 = conv_block(int_ch+int_ch,int_ch,rate=2)

    self.conv_8 = conv_block(int_ch+int_ch,out_ch,rate=1)

  def forward(self,x):
      
    
    x0 = self.conv_1(x)

    """Encoder"""
    x1 = self.conv_2(x0)
    x2 = self.conv_3(x1)
    x3 = self.conv_4(x2)

    """Bridge"""
    x4 = self.conv_5(x3) # [2, 256, 32, 32]

    """Decoder"""
    # print(x4.shape,x3.shape)  # [2, 256, 32, 32]) [2, 256, 32, 32]
    # print(x.shape,x2.shape) # [2, 512, 32, 32] [2, 256, 32, 32]
    # print(x.shape,x1.shape) # [2, 512, 32, 32] [2, 256, 32, 32]

    x = torch.cat([x4,x3],axis=1)
    x = self.conv_6(x) # [2, 256, 32, 32]
    
    x  = torch.cat([x,x2],axis=1)
    x = self.conv_7(x) # [2, 256, 32, 32]
    

    x = torch.cat([x,x1],axis=1)
    x = self.conv_8(x) # [2, 512, 32, 32]

    return x+x0

class u2net(nn.Module):
  def __init__(self,ch_in,ch_out,ch_int,num_classes=2):
    super().__init__()

    self.rsul_7 = RSU_L(ch_in,ch_out[0],ch_int[0],7)
    self.maxPool = nn.MaxPool2d(kernel_size=2)

    self.rsul_6 = RSU_L(ch_out[0],ch_out[1],ch_int[1],6)


    self.rsul_5 = RSU_L(ch_out[1],ch_out[2],ch_int[2],5)

    self.rsul_4 = RSU_L(ch_out[2],ch_out[3],ch_int[3],4)


    self.rsu4f_1 = RSU_4F(ch_out[3],ch_out[4],ch_int[4])


    self.rsu4f_2 = RSU_4F(ch_out[4],ch_out[5],ch_int[5])

    self.up_1 = nn.Upsample(scale_factor=2,mode="bilinear")

    self.rsu4f_3 = RSU_4F(ch_out[5]+ch_out[5],ch_out[6],ch_int[6])

    self.rsul_de_4 = RSU_L(ch_out[6]+ch_out[6],ch_out[7],ch_int[7],4)

    self.rsul_de_5 = RSU_L(ch_out[7]+ch_out[7],ch_out[8],ch_int[8],5)

    self.rsul_de_6 = RSU_L(ch_out[8]+ch_out[8],ch_out[9],ch_int[9],6)

    self.rsul_de_7 = RSU_L(ch_out[9]+ch_out[9],ch_out[10],ch_int[10],7)

    self.side_1_conv = nn.Conv2d(ch_out[10],num_classes,kernel_size=3,
                                 padding="same")
    
    self.side_2_conv = nn.Conv2d(ch_out[9],num_classes,kernel_size=3,
                                 padding="same")
    
    self.side_3_conv = nn.Conv2d(ch_out[8],num_classes,kernel_size=3,
                                 padding="same")
    
    self.up_2 = nn.Upsample(scale_factor=4,mode="bilinear")


    self.side_4_conv = nn.Conv2d(ch_out[7],num_classes,kernel_size=3,
                                 padding="same")
    

    self.up_3 = nn.Upsample(scale_factor=8,mode="bilinear")


    self.side_5_conv = nn.Conv2d(ch_out[6],num_classes,kernel_size=3,
                                 padding="same")
    self.up_4 = nn.Upsample(scale_factor=16,mode="bilinear")

    self.side_6_conv = nn.Conv2d(ch_out[5],num_classes,kernel_size=3,
                                 padding="same")
    
    self.up_5 = nn.Upsample(scale_factor=32,mode="bilinear")

    self.final_conv = nn.Conv2d(num_classes*6,num_classes,kernel_size=3,
                                 padding="same")

  def forward(self,x):
    s1 = self.rsul_7(x) # [2, 64, 512, 512]
    p1 = self.maxPool(s1) # [2, 64, 256, 256]
    
    print(p1.shape)

    # s2 = self.rsul_6(p1) #[2, 128, 256, 256]
    # p2 = self.maxPool(s2) # [2, 128, 128, 128]

    # s3 = self.rsul_5(p2) #[2, 256, 128, 128]
    # p3 = self.maxPool(s3) # [2, 256, 64, 64]

    # s4 = self.rsul_4(p3) # [2, 512, 64, 64]
    # p4 = self.maxPool(s4) # [2, 512, 32, 32]

    # s5 = self.rsu4f_1(p4) # [2, 512, 32, 32]
    # p5 = self.maxPool(s5) # [2,512,16,16]

    # # Bridge
    # b1 = self.rsu4f_2(p5) # [2, 512, 16, 16]
    # b2 = self.up_1(b1) #  [2, 512, 32, 32]

    # # Decoder
    # d1 = torch.cat([b2,s5],axis=1) #[2, 1024, 32, 32]
    # d1 = self.rsu4f_3(d1) #[2, 512, 32, 32]
    # u1 = self.up_1(d1)  # [2, 512, 64, 64]

    # d2 = torch.cat([u1,s4],axis=1) #[2, 1024, 64, 64]
    # d2 = self.rsul_de_4(d2) # [2, 256, 64, 64]
    # u2 = self.up_1(d2) # [2, 256, 128, 128]


    # d3 = torch.cat([u2,s3],axis=1) #[2, 512, 128, 128]
    # d3 = self.rsul_de_5(d3) # [2, 128, 128, 128]
    # u3 = self.up_1(d3) # [2, 128, 256, 256]


    # d4 = torch.cat([u3,s2],axis=1) #[2, 256, 256, 256]
    # d4 = self.rsul_de_6 (d4) # [2, 64, 256, 256]
    # u4 = self.up_1(d4) # [2, 64, 512, 512]


    # d5= torch.cat([u4,s1],axis=1) #[2, 128, 512, 512]
    # d5 = self.rsul_de_7(d5) # [2, 64, 512, 512]
    
    
    # # Side Outputs
    # y1 = self.side_1_conv(d5) # [2, 2, 512, 512]
    
    # y2 = self.side_2_conv(d4) # [2, 2, 256, 256]
    # y2 = self.up_1(y2) # [2, 2, 512, 512]


    # y3 = self.side_3_conv(d3) # [2, 2, 128, 128]
    # y3 = self.up_2(y3) # [2, 2, 512, 512]
    
    # y4 = self.side_4_conv(d2) # [2, 2, 64, 64]
    # y4 = self.up_3(y4) # [2, 2, 512, 512]
    
    # y5 = self.side_5_conv(d1) # [2, 2, 32, 32]
    # y5 = self.up_4(y5) # [2, 2, 512, 512]

    # y6 = self.side_6_conv (b1) # [2, 2, 16, 16]
    # y6 = self.up_5(y6) # [2, 2, 512, 512]


    
    # y0 = torch.cat([y1,y2,y3,y4,y5,y6],axis=1) # [2, 12, 512, 512]
    # y0 = self.final_conv(y0) # [2, 2, 512, 512]

    # return  y0,y1,y2,y3,y4,y5,y6

if __name__ == "__main__":
    
    
    t1 = torch.zeros((2,3,512,512), device=torch.device('cuda'))
    ou_ch = [64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]
    in_ch = [32, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16]
    model = u2net(ch_in=3, ch_out=ou_ch, ch_int=in_ch)
    model = model.to("cuda:0")
    
    model(t1)
    
    # print(next(model.parameters()).device)
    # print(t1.dtype)
    
    # y0,y1,y2,y3,y4,y5,y6 = model(t1)
    
    # ConvBlock
    # cnv_block = conv_block(3,64)
    # cnv_block(t1)
    
    # RSU_L
    rsu_l = RSU_L(3,64,64,7)
    rsu_l(t1)
    
    # RSU_R

    
    
    
    