import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ResidualBlocks import ResidualBlock
from Attention import Attention

class UNet_Block(nn.Module):
    def __init__(self,upscale:bool,attention:bool,C:int,num_groups:int,dropout_prob:float,num_heads:int):
        super().__init__()
        self.resBlock1=ResidualBlock(C=C,num_groups=num_groups,dropout_prob=dropout_prob)
        self.resBlock2=ResidualBlock(C=C,num_groups=num_groups,dropout_prob=dropout_prob)

        if upscale:
            self.conv=nn.ConvTranspose2d(in_channels=C,out_channels=C//2,kernel_size=4,stride=2,padding=1)
        else:
            self.conv=nn.Conv2d(in_channels=C,out_channels=C*2,kernel_size=3,stride=2,padding=1)
        
        if attention:
            self.attention=Attention(C=C,num_heads=num_heads,dropout_prob=dropout_prob)

    def forward(self,x,embeddings):
        x=self.resBlock1(x,embeddings)
        if hasattr(self,'attention'):
            x=self.attention(x)
        x=self.resBlock2(x,embeddings)
        return self.conv(x),x
    
if __name__=="__main__":
    from SinusoidalEmbeddings import SinusoidalEmbeddings

    embeddings=SinusoidalEmbeddings(time_steps=1000,embed_dim=512)
    x=torch.randn(1,384,64,198)
    embed=embeddings(x,[100])
    model=UNet_Block(True,False,384,32,0.1,8)
    output,residual=model(x,embed)
    print(output.shape)
    print(residual.shape)
    print(output)
    print(residual)
