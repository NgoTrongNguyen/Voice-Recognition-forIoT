import torch
import torch.nn as nn
from Data import SpeechDataset
from torch.utils.data import DataLoader

# Input Definition

data_dir = "Speech\\"
dataset = SpeechDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
num_classes = 5
input_size= 200

# Model

class SpeechCNN(nn.Module):
    def __init__(self):
        super(SpeechCNN, self).__init__()

        # Conv1        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), padding = 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2))

        # Conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2))

        # Fully_Connected
        self.fc1 = nn.Linear(self._get_conv_output(input_size), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)


    def _get_conv_output(self, input_size):

        # Size
        x = torch.zeros(1, 1, 40, input_size)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))    
        return int(torch.prod(torch.tensor(x.shape[1:]))) 

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

