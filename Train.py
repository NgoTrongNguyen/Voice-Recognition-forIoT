from Model import SpeechCNN
from Data import SpeechDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# Input Definition

NUM_EPOCHS = 100

data_dir = "Speech\\"
dataset = SpeechDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
num_classes = 5
some_output_size = 1

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
    torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
