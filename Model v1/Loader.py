from Model import SpeechCNN
from Data import SpeechDataset
from torch.utils.data import DataLoader
import torch

checkpoint = torch.load("checkpoint_epoch_41.pth")  
model = SpeechCNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  

test_dataset = SpeechDataset("TestSpeech\\")

for i in range(4):
    input_tensor, label = test_dataset[i]   
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0) 

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    print(f"Predicted label: {predicted.item()}")
    