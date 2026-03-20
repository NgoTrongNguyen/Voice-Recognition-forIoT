import librosa
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch



class Speeching(Dataset):
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
    def pad_or_truncate(data, max_len=30):
        if data.shape[1] < max_len:
            pad_width = max_len - data.shape[1]
            data = np.pad(data, ((0,0),(0,pad_width)), mode='constant')
        else:
            data = data[:, :max_len]
        return data

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        audio, sr = librosa.load(file_path)

        audio = self.add_noise(audio, 0.025)

        melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=80)
        melspec = self.pad_or_truncate(melspec, max_len=200)
        melspec_tensor = torch.tensor(melspec, dtype=torch.float32)

        label = torch.tensor(int(self.file_list[idx].split('_')[0]), dtype=torch.long)
   
        return melspec_tensor, label

    

data_dir = "Speech\\"
dataset = Speeching(data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    melspec, labels = batch
    print("Batch MFCCs shape:", melspec)
    print("Batch labels:", labels)
    break
