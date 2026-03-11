import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import numpy as np

 
class SpeechDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
 
    def __len__(self):
        return len(self.file_list)
 
    def add_noise(audio, noise_factor):
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio
 
    def __getitem__(self, idx):
        # Loading audio from Folder

        file_path = os.path.join(self.data_dir, self.file_list[idx])
        audio, sr = librosa.load(file_path)

        audio = self.add_noise(audio, 0.08)

        # Converting to tensor
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_tensor = torch.tensor(mfccs, dtype=torch.float32)

        label = int(self.file_list[idx].split('_')[0])
        return mfcc_tensor, label
 
 
# Input Definition

data_dir = "Speech\\"
dataset = SpeechDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
 