import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self,C:int,num_groups:int,dropout_prob:float):
        super().__init__()

        self.relu=nn.ReLU(inplace=True)
        self.group1=nn.GroupNorm(num_groups=num_groups,num_channels=C)
        self.group2=nn.GroupNorm(num_groups=num_groups,num_channels=C)

        self.conv1=nn.Conv2d(in_channels=C,out_channels=C,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=C,out_channels=C,kernel_size=3,padding=1)

        self.dropout=nn.Dropout(p=dropout_prob,inplace=True)
    
    def forward(self,x,embeddings):
        x=x+embeddings[:x.shape[0],:x.shape[1],:,:]
        #print(x)

        r=self.conv1(self.relu(self.group1(x)))
        #print(r.shape)
        r=self.dropout(r)
        #print(r.shape)
        r=self.conv2(self.relu(self.group2(r)))
        #print(r.shape)

        #print(r+x)

        return r+x

if __name__=="__main__":
    from SinusoidalEmbeddings import SinusoidalEmbeddings

    embeddings=SinusoidalEmbeddings(time_steps=1000,embed_dim=512)
    x=torch.randn(1,64,64,198)
    embed=embeddings(x,[100])
    model=ResidualBlock(64,32,0.1)
    output=model(x,embed)
    print(output.shape) #torch.Size([1,64,64,198])
