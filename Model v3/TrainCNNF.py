from Conformer import ConformerSTT
from DataCNNF import loader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Cấu hình
d_model = 256
n_head = 4
batch_size = 8
dropout = 0.1
n_mels=80
num_classes=30
num_blocks=12
time_steps = 100
kernel_size = 31
lr = 1e-4
weight_decay = 1e-6
EPOCHS = 10

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConformerSTT(n_mels, num_classes).to(device)


def train_loop(batch):
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay)

    for epoch in range(EPOCHS):
        mels, targets, mel_lens, target_lens = batch 
        mels, targets = mels.to(device), targets.to(device)

        logits = model(mels) 
    
        logits = logits.transpose(0, 1)
        log_probs = F.log_softmax(logits, dim=-1)
    
        input_lengths = mel_lens // 4 

        loss = criterion(log_probs, targets, input_lengths, target_lens)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03} | Loss: {loss/len(batch):.4f}")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'd_model': d_model,
                'n_head': n_head,
                'dropout': dropout,
                'n_mels': n_mels,
                'num_classes': num_classes,
                'num_blocks': num_blocks
            }
            torch.save(checkpoint, f"epochCNNF{epoch+1}.ckpt")
    
    return loss.item()

data = loader()
train_loop(data)
output = model(data)
