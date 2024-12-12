import os
import torch
import torchaudio
from torch.utils.data import Dataset

class FeatureExtractor(Dataset):
    def __init__(self,audio_path,tensor_path,melspectrogram_transform,target_sample_rate,device):
        self.audio_path=audio_path
        self.tensor_path=tensor_path
        self.melspectrogram_transform=melspectrogram_transform.to(device)
        self.target_sample_rate=target_sample_rate
        self.device=device
    
    #resample audio signal to the target sample rate
    def _resample_if_req(self,signal,sr):
        if sr!=self.target_sample_rate:
            resampler=torchaudio.transforms.Resample(sr,self.target_sample_rate).to(self.device)
            signal=resampler(signal)
        return signal
    
    #if the audio has multiple channels mix it down by taking the average(mean)
    def _mix_down_if_req(self,signal):
        if signal.shape[0]>1:
            signal=torch.mean(signal,dim=0,keepdim=True)
        return signal
    
    def _get_audio_file_path(self,idx):
        return self.audio_path+os.listdir(self.audio_path)[idx]

    def __len__(self):
        return len(os.listdir(self.audio_path))
    
    def __getitem__(self,idx):
        file_path=self._get_audio_file_path(idx)

        #load audio signal from audio file
        signal,sr=torchaudio.load(file_path)
        signal=signal.to(self.device)
        
        #preprocess signal
        signal=self._resample_if_req(signal,sr)
        signal=self._mix_down_if_req(signal)   #not done for beacuse dont know the type of input diffusion models take

        #apply transformations (MelSpectrogram)
        signal=self.melspectrogram_transform(signal)

        #save the resultant tensor on the disk
        for i in range(3):
            signal1=signal[:,:,66*i:66*(i+1)]
            torch.save(signal1,self.tensor_path + os.listdir(self.audio_path)[idx][:-4]+f"_{i+1}.pt")

        #return the filename of the saved tensor
        return self.tensor_path + os.listdir(self.audio_path)[idx][:-4]+f"_{i+1}.pt",signal #os.listdir(self.audio_path)[idx][:-3]+"pt",signal





if __name__=="__main__":
    audio_path="/home/vivek/AML_Project/Dataset/"
    tensor_path="/home/vivek/AML_Project/Dataset_Tensor_3part/"
    SAMPLE_RATE=10000
    melspectrogram=torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=64,f_max=5000,power=2.0) #n_fft: Window size of samples for each Short-Time Fourier Transform (STFT), hop_length: should be half or 1/4th of n_fft, keeping it halp of n_fft ensures that there is 50% overlap between adjacent windows.f_max : Maximum frequency (Nyquist frequency), power: Power Spectrogram, Setting this to 2 computes a power spectrogram, which is commonly used in machine learning tasks.

    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    
    print(f"Using device {device}")

    feature_dataset=FeatureExtractor(audio_path,tensor_path,melspectrogram,SAMPLE_RATE,device)

    for filename,signal in feature_dataset:
        print(filename)
        print(signal.shape)