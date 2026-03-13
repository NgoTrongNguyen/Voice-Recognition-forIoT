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

    @staticmethod
    def add_noise(audio, noise_factor):
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio
    
    @staticmethod
    def pad_or_truncate(mfcc, max_len=200):
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
 
    def __getitem__(self, idx):
        # Loading audio from Folder

        file_path = os.path.join(self.data_dir, self.file_list[idx])
        audio, sr = librosa.load(file_path)

        audio = self.add_noise(audio, 0.025)

        # Converting to tensor
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs = self.pad_or_truncate(mfccs, max_len=200)
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-6)
        mfcc_tensor = torch.tensor(mfccs, dtype=torch.float32)

        label = torch.tensor(int(self.file_list[idx].split('_')[0]), dtype=torch.long)
        return mfcc_tensor, label
 
 
# Input Definition

data_dir = "Speech\\"
dataset = SpeechDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

"""
for batch in dataloader:
    mfccs, labels = batch
    print("Batch MFCCs shape:", mfccs)
    print("Batch labels:", labels)
    break

 """