import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class SinusoidalEmbeddings(nn.Module):
    def __init__(self,time_steps:int,embed_dim:int):
        super().__init__()

        position=torch.arange(time_steps).unsqueeze(1).float()

        div=torch.exp(torch.arange(0,embed_dim,2).float() * -(math.log(10000.0)/embed_dim))

        embeddings=torch.zeros(time_steps,embed_dim,requires_grad=False)

        embeddings[:,0::2]=torch.sin(position * div)

        embeddings[:,1::2]=torch.cos(position * div)

        self.embeddings=embeddings
    
    def forward(self,x,t):
        embeds=self.embeddings[t].to(x.device)
        
        return embeds[:,:,None,None]

if __name__=="__main__":
    x=torch.randn(1,1,64,198)
    embeddings=SinusoidalEmbeddings(time_steps=1000,embed_dim=512)
    embed=embeddings(x,[100])

    print(embed.shape)  #torch.Size([1, 512, 1, 1])