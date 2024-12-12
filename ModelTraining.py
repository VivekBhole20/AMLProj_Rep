import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from timm.utils import ModelEmaV3
from tqdm import tqdm
import math
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from typing import List
from AudioLoader import AudioLoader
from DDPM_Scheduler import DDPM_Scheduler
from UNET import UNET

def set_seed(seed:int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(batch_size:int=1,num_time_steps:int=1000,num_epochs:int=15,seed:int=-1,ema_decay:float=0.9999,lr=2e-5,checkpoint_path:str=None):
    set_seed(random.randint(0,2**32-1)) if seed == -1 else set_seed(seed)

    tensor_path="/home/vivek/AML_Project/Dataset_Tensor/"
    CSV_file_path="/home/vivek/AML_Project/musiccaps-public_with_filename.csv"

    if(torch.cuda.is_available()):
        device="cuda"
    else:
        device="cpu"
    
    print(f"Using device: {device}")

    #making the dataset
    train_dataset=AudioLoader(tensor_path,CSV_file_path,device,200)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    scheduler=DDPM_Scheduler(num_time_steps=num_time_steps)
    model=UNET().to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    ema=ModelEmaV3(model,decay=ema_decay)   #Exponential Moving average of weights

    if checkpoint_path is not None:
        checkpoint=torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion=nn.MSELoss(reduction='mean')

    patience=4
    best_loss=1
    stop_loss=0

    for i in range(num_epochs):
        total_loss=0
        for bidx, x in enumerate(tqdm(train_dataloader,desc=f"Epoch {i+1}/{num_epochs}")):
            t=torch.randint(0,num_time_steps,(batch_size,))
            e=torch.randn_like(x,requires_grad=False)
            a=scheduler.alpha[t].view(batch_size,1,1,1).to(device)
            x=(torch.sqrt(a[:x.shape[0]])*x)+(torch.sqrt(1-a[:x.shape[0]])*e)
            with torch.cuda.amp.autocast():
                output=model(x,t)
                #print(output)
                optimizer.zero_grad()
                loss=criterion(output,e)
            total_loss+=loss.item()
            loss.backward()
            #max-norm regularization
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            ema.update(model)
        print(f"Total Loss : {total_loss}")
        epoch_loss=total_loss / 5356
        print(f'Epoch {i+1} | Loss {epoch_loss}')

        # early stopping
        if best_loss>epoch_loss:
            best_loss=epoch_loss
            checkpoint={
                'weights':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'ema':ema.state_dict()
            }
        else:
            stop_loss=stop_loss+1
        
        if stop_loss==patience:
            break

    torch.save(checkpoint,'checkpoints/ddpm_checkpoint')


def melspec_to_audio(melspec,sample_rate,path):
    print(melspec)
    if torch.isnan(melspec).any() or torch.isinf(melspec).any():
        raise ValueError("Input mel spectrogram contains NaN or Inf values.")
    print("Shape of output:", (melspec).shape)
    
    inverse_melscale_transform = torchaudio.transforms.InverseMelScale(
    n_stft=1024//2+1,
    n_mels=128,
    sample_rate=sample_rate,
    f_min=0.0,
    f_max=sample_rate//2
    ).to("cpu")
    print("Shape of filter bank:", inverse_melscale_transform.fb.shape)
    melspec=melspec.to("cpu")
    linear_spectrogram = inverse_melscale_transform(melspec)

    griffin_lim_transform = torchaudio.transforms.GriffinLim(
    n_fft=1024,
    hop_length=512
    )
    reconstructed_waveform = griffin_lim_transform(linear_spectrogram)

    torchaudio.save(path,reconstructed_waveform,sample_rate)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    #plt.title("Original Waveform")
    #plt.plot(waveform.t().numpy())
    #plt.subplot(1, 2, 2)
    plt.title("Reconstructed Waveform")
    plt.plot(reconstructed_waveform.t().numpy())
    plt.tight_layout()
    plt.show()


def display_reverse(images:List):
    for i in range(len(images)):
        x = images[i].squeeze(0)
        melspec_to_audio(x,10000,f"generated_music/128Mels/gen_audio_{i+1}.wav")
        # x = rearrange(x, 'c h w -> h w c')
        # x = x.numpy()
        # ax.imshow(x)
        # ax.axis('off')

def inference(checkpoint_path: str=None,
              num_time_steps: int=1000,
              ema_decay: float=0.9999, ):
    if(torch.cuda.is_available()):
        device="cuda"
    else:
        device="cpu"
    
    print(f"Using device: {device}")
    checkpoint=torch.load(checkpoint_path)
    model=UNET().to(device)
    model.load_state_dict(checkpoint['weights'])
    ema=ModelEmaV3(model,decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler=DDPM_Scheduler(num_time_steps=num_time_steps)
    times=[0,15,50,100,200,300,400,550,700,999]
    images=[]

    with torch.no_grad():
        model=ema.module.eval()
        for i in range(10):
            z=torch.randn(1,1,128,200)

            for t in reversed(range(1,num_time_steps)):
                t=[t]
                noise_scale=( scheduler.beta[t]/((torch.sqrt(1-scheduler.alpha[t])) * (torch.sqrt(1-scheduler.beta[t]))))
                z=(1/(torch.sqrt(1-scheduler.beta[t])))*z-(noise_scale*model(z.to(device),t).cpu())
                #print(f"gaussian Z: {model(z.to(device),t).cpu()}")
                #exit()
                #if t[0] in times:
                #    images.append(z)
                e=torch.randn(1,1,128,200)
                z=z+(e*torch.sqrt(scheduler.beta[t]))
            
            final_noise_scale=scheduler.beta[0]/((torch.sqrt(1-scheduler.alpha[0])) * (torch.sqrt(1-scheduler.beta[0])))
            x=(1/(torch.sqrt(1-scheduler.beta[0])))*z-(final_noise_scale*model(z.to(device),[0]).cpu())

            print(f"Image: {i+1}")
            images.append(x)
            display_reverse(images)
            #x=rearrange(x.squeeze(0),'c h w','h w c').detach()

def main():
    #train()
    inference('checkpoints/ddpm_checkpoint')

if __name__=="__main__":
    main()


