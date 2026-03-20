# embedding + positional encoding + vài lớp self-attention (causal) + MLP +
# head dự đoán token tiếp theo.

import math
import torch
import torch.nn as nn
import os

# Attention kèm kVCache
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)
        #Chance to make some to 0
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, past_kv = None, use_cache = False): 
        # For training mode: Use None and False
        #B: batch size, T: seq length, C: d_model
        B,T,C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)

        q = qkv[:, :, 0].transpose(1, 2)  # B, H, T, D
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        curr_T = k.size(2)
        mask = torch.tril(torch.ones(curr_T, curr_T, device=x.device))[-T:, :]
        att = att.masked_fill(mask == 0, float("-inf"))
        #Chuẩn hóa ma trận attention
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # B, heads, T, head_dim
        y = y.transpose(1,2).contiguous().view(B, T, C)

        new_kv = (k, v) if use_cache else None
        return self.out(y), new_kv

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, past_kv = None, use_cache = False):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv
    
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, dropout, block_size):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, past_kvs = None, use_cache = False):
        B, T = x.shape

        tok_emb = self.token_emb(x)
        
        if past_kvs is None:
            pos = torch.arange(0, T, device=x.device).unsqueeze(0) # [0,1,...,T-1]
        else:
            past_len = past_kvs[0][0].size(2)
            pos = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(0)

        pos_emb = self.pos_emb(pos) # (T, d_model)
        x = tok_emb + pos_emb # B, T, d_model

        new_past = []

        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, past_kv, use_cache) # B, T, d_model
            if use_cache:
                new_past.append(kv)

        x = self.ln_f(x) # B, T, d_model
        logits = self.head(x) # B, T, vocab_size

        return logits, new_past if use_cache else None
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature, top_k, eos_token_id=2):
        self.eval()
        past = None

        for i in range(max_new_tokens):
            # Cắt chuỗi khi vượt size
            logits, past = self.forward(
                idx[:, -1:], past_kvs=past, use_cache=True
            )
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            # Lấy token cuối

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            # Sampling chọn token mới
            next_id = torch.multinomial(probs, num_samples=1)
            # Ghép token mới vào chuỗi
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == eos_token_id:
                break
        return idx

class RobotTokenizer:
    def __init__(self):
        # Thêm sẵn các nhãn điều khiển vào vocab để không bị UNK
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "<SEP>": 4, 
                     "MOVE_F": 5, "MOVE_B": 6, "TURN_L": 7, "TURN_R": 8, "STOP": 9, "TURN_AROUND": 10}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def fit(self, sentences):
        for line in sentences:
            # Tách từ nhưng giữ nguyên các nhãn đặc biệt
            words = line.replace("<SEP>", " <SEP> ").replace("<EOS>", " <EOS> ").split()
            for word in words:
                if word not in self.stoi:
                    self.stoi[word] = self.vocab_size
                    self.itos[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text, add_special=True):
        tokens = [self.stoi.get(w, self.stoi["<UNK>"]) for w in text.split()]
        if add_special:
            tokens = [self.stoi["<SOS>"]] + tokens + [self.stoi["<EOS>"]]
        return tokens

    def decode(self, ids):
        return " ".join([self.itos[i] for i in ids if i > 2])
    
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, tokenizer, data_pairs, epochs=100, batch_size=16, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 1. Chuẩn bị dữ liệu
    tokenizer.fit(data_pairs)
    encoded_data = [tokenizer.encode(p) for p in data_pairs]
    
    # Padding cho đồng nhất kích thước
    max_len = max(len(d) for d in encoded_data)
    padded_data = [d + [0] * (max_len - len(d)) for d in encoded_data]
    
    X = torch.tensor([d[:-1] for d in padded_data], dtype=torch.long)
    Y = torch.tensor([d[1:] for d in padded_data], dtype=torch.long)
    
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Bỏ qua PAD token khi tính loss

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # Reset cache mỗi batch huấn luyện (không dùng cache khi train)
            logits, _ = model(x, past_kvs=None, use_cache=False)
            
            # Reshape để tính loss: (B*T, Vocab)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
            torch.save({
                "model": model.state_dict(),
                "block_size": 64,
                "d_model": 128,
                "n_layers": 4,
                "n_heads": 4,
                "dropout": 0.15
            }, f"epoch{epoch+1}.pt")

    print("Training hoàn tất!")
    return model

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    return lines

data = load_dataset("corpus.txt")

tokenizer = RobotTokenizer()
tokenizer.fit(data)

model = GPT(vocab_size=tokenizer.vocab_size, d_model=128, n_layers=4, n_heads=4, dropout=0.1, block_size=64)

# Huấn luyện
trained_model = train_model(model, tokenizer, data)

# Thử nghiệm dự đoán với KV Cache
prompt = input() + "<SEP>"
input_ids = torch.tensor([tokenizer.encode(prompt, add_special=False)]).to(next(model.parameters()).device)

# Generate tiếp 3 từ tiếp theo
output_ids = model.generate(input_ids, max_new_tokens=15, temperature=0.01, top_k=1)
output = tokenizer.decode(output_ids[0].tolist())
find = output.find("<SEP>")
print("Kết quả dự đoán: ", output[find:])