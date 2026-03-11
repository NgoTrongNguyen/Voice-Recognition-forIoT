from Model import SpeechCNN
from Data import SpeechDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from Train import SpeechCNN

model = SpeechCNN()

test_dataset = SpeechDataset('your_test_data_directory')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print(f'Accuracy: {100 * correct / total}%')