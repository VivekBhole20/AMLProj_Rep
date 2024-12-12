import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset,DataLoader

class AudioLoader(Dataset):
    def __init__(self,tensor_path,CSV_file_path,device,num_samples):
        self.tensor_path=tensor_path
        self.CSV_file_path=CSV_file_path
        self.text_data=pd.read_csv(self.CSV_file_path)
        self.device=device
        self.num_samples=num_samples
    
    def _mix_down_if_req(self,signal):
        if signal.shape[0]>1:
            signal=torch.mean(signal,dim=0,keepdim=True)
        return signal
    
    def _right_padding_if_req(self,signal):
        signal_len=signal.shape[2]
        if signal_len<self.num_samples:
            missing_pad=self.num_samples-signal_len
            signal=torch.nn.functional.pad(signal,(0,missing_pad),"constant",1e-6)
        return signal


    def _get_tensor_file_path(self,idx):
        return self.tensor_path + os.listdir(self.tensor_path)[idx]

    def __len__(self):
        return len(os.listdir(self.tensor_path))
    
    def __getitem__(self,idx):
        tensor_file_path=self._get_tensor_file_path(idx)

        #load the tensor file
        tensor=torch.load(tensor_file_path,map_location=self.device,weights_only=True)

        tensor=self._mix_down_if_req(tensor)

        
        tensor=self._right_padding_if_req(tensor)

        tensor=torch.log(tensor + 1e-6)

        #extract the file name of the tensor
        file_name=tensor_file_path.split('/')[-1][:-2]+"wav"#.split('.')[0].split('\u00FE')[0]

        #extract the text data from the CSV file for this audio file
        data_row=self.text_data[self.text_data['filenames']=="Dataset/" + file_name]

        return tensor #,data_row.to_dict()


if __name__ == "__main__":
    tensor_path="/home/vivek/AML_Project/Dataset_Tensor_3part/"
    CSV_file_path="/home/vivek/AML_Project/musiccaps-public_with_filename.csv"
    num_samples=68 # after melspectrogram all tensors should have dimensions (1,128,198)

    if(torch.cuda.is_available()):
        device="cuda"
    else:
        device="cpu"

    print(f"Using device: {device}")

    #making the dataset
    dataset=AudioLoader(tensor_path,CSV_file_path,device,num_samples=num_samples) 

    print("Dataset done")

    count=0
    error=False
    for tensor in dataset:
        #print(data)
        print(tensor.shape)
        print(tensor)
        if tensor.shape!=([1,64,198]):
            error=True
        count+=1
    
    print(count)
    print(error)

    # giving dataset to dataloader
    torch.manual_seed(1)
    dataloader=DataLoader(dataset,batch_size=20,shuffle=True)

    for x in dataloader:
        print(x.shape)

    print("DataLoader done")