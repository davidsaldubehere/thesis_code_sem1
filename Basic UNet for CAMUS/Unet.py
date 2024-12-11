"""
Base on tutorial :https://towardsdev.com/original-u-net-in-pytorch-ebe7bb705cc7
Implementation Unet
"""

import torch
import torch.nn as nn
from DownConvolution import DownConvolution
from LastConvolution import LastConvolution


from SimpleConvolution import SimpleConvolution
from UpConvolution import UpConvolution


class Unet(nn.Module):
    def __init__(self,input_channel,num_classes,light=False):
        super(Unet,self).__init__()
        if not light:        
            #PAPER UNET (31M parameters)
            #Encoder Part
            self.simpleConv = SimpleConvolution(input_channel,64)
            self.downBlock1 = DownConvolution(64,128)
            self.downBlock2 = DownConvolution(128,256)
            self.downBlock3 = DownConvolution(256,512)
    
            #Last level of Unet
            self.midmaxpool = nn.MaxPool2d(2,2)
            self.bridge = UpConvolution(512,1024)
    
            #Decoder Part
            self.upBlock1 = UpConvolution(1024,512)
            self.upBlock2 = UpConvolution(512,256)
            self.upBlock3 = UpConvolution(256,128)
            self.lastConv = LastConvolution(128,64,num_classes)
        else:
            #SIMPLIFIED UNET (3.7M parameters)
            self.simpleConv = SimpleConvolution(input_channel,22)
            self.downBlock1 = DownConvolution(22,44)
            self.downBlock2 = DownConvolution(44,88)
            self.downBlock3 = DownConvolution(88,176)
    
            self.midmaxpool = nn.MaxPool2d(2,2)
            self.bridge = UpConvolution(176,352)
    
            self.upBlock1 = UpConvolution(352,176)
            self.upBlock2 = UpConvolution(176,88)
            self.upBlock3 = UpConvolution(88,44)
            self.lastConv = LastConvolution(44,22,num_classes)

    def forward(self,x):
        x_1 = self.simpleConv(x)
        x_2 = self.downBlock1(x_1)

        x_3 = self.downBlock2(x_2)
        x_4 = self.downBlock3(x_3)

        x_5 = self.midmaxpool(x_4)

        x_6 = self.bridge(x_5)

        x_4_6 = torch.cat((x_4,x_6),1)

        x_7 = self.upBlock1(x_4_6)
         
        x_3_7 = torch.cat((x_3,x_7),1)

        x_8 = self.upBlock2(x_3_7)

        x_2_8 = torch.cat((x_2,x_8),1)
        x_9 = self.upBlock3(x_2_8)

        x_1_9 = torch.concat((x_1,x_9),1)
        out = self.lastConv(x_1_9)

        return out
    
    
def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")