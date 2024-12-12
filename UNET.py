import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from SinusoidalEmbeddings import SinusoidalEmbeddings
from UNet_Block import UNet_Block
from typing import List

class UNET(nn.Module):
    def __init__(self,
                 Channels: List = [64, 128, 256, 512, 512, 384],
                 Attentions: List = [False,True,False,False,False,True],
                 Upscales: List = [False,False,False,True,True,True],
                 num_groups:int = 32,
                 dropout_prob:float = 0.1,
                 num_heads:int = 8,
                 input_channels:int = 1,
                 output_channels:int = 1,
                 time_steps:int=1000
                 ):
        super().__init__()
        self.num_layers=len(Channels)
        self.shallow_conv=nn.Conv2d(in_channels=input_channels,out_channels=Channels[0],kernel_size=3,padding=1)
        out_channels=(Channels[-1]//2)+Channels[0]
        self.late_conv=nn.Conv2d(in_channels=out_channels,out_channels=out_channels//2,kernel_size=3,padding=1)
        self.output_conv=nn.Conv2d(in_channels=out_channels//2,out_channels=output_channels,kernel_size=1)
        self.relu=nn.ReLU(inplace=True)
        self.embeddings=SinusoidalEmbeddings(time_steps=time_steps,embed_dim=max(Channels))

        for i in range(self.num_layers):
            layer=UNet_Block(
                upscale=Upscales[i],
                attention=Attentions[i],
                C=Channels[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                num_heads=num_heads
            )

            setattr(self,f'Layer{i+1}',layer)
        
    
    def forward(self,x,t):
        x=self.shallow_conv(x) #torch.Size([1, 64, 64, 198])
        #print(x.shape)

        residuals=[]

        for i in range(self.num_layers//2):
            layer=getattr(self,f'Layer{i+1}')
            embeddings=self.embeddings(x,t)
            x,r=layer(x,embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2,self.num_layers):
            layer=getattr(self,f'Layer{i+1}')
            #print("Shape of layer output:", (layer(x, embeddings))[0].shape)
            #print("Shape of residuals:", residuals[self.num_layers - i - 1].shape)
            x=torch.concat((layer(x,embeddings)[0],residuals[self.num_layers-i-1]),dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))
    
if __name__=="__main__":
    x=torch.randn(1,1,64,200)
    model=UNET()
    output=model(x,[999])
    print(output)