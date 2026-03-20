from Model import SpeechCNN
from Data import SpeechDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# Input Definition

NUM_EPOCHS = 45

data_dir = "Speech\\"
dataset = SpeechDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
num_classes = 5
some_output_size = 200

# Model
model = SpeechCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)

# Training

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

    # Checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }
    if (epoch>148):
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')


test_dataset = SpeechDataset("TestSpeech\\")
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