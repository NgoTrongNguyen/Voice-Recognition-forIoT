import torch
import torch.nn as nn
import math

# Conv2d
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU()
        )
    def forward(self, x):
        # (B, C, T, F)
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        return self.out(x)

# Multihead Attention
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.fully_connected = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        # (batch, seq_len, d_model) -> (batch, seq_len, n_head, d_k) 
        q = q.view(batch_size, -1, self.n_head, self.d_k)   
        k = k.view(batch_size, -1, self.n_head, self.d_k)   
        v = v.view(batch_size, -1, self.n_head, self.d_k)

        # (batch, seq_len, n_head, d_k) -> (batch, n_head, seq_len, d_k)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        attn = (torch.matmul(q,k.transpose(-2,-1)))/math.sqrt(self.d_k)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        attn = torch.matmul(attn, v)

        # (batch, n_head, seq_len, d_k) -> (batch, seq_len, n_head, d_k) -> (batch, seq_len, d_model)
        out = attn.transpose(1, 2)
        out = out.contiguous().view(batch_size, -1, self.d_model)

        return self.fully_connected(out)

# Conformer
class Conformer(nn.Module):
    def __init__(self, d_model, n_head, kernel_size, dropout):
        super(Conformer, self).__init__()

        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout)
        )
        
        self.attn = MultiheadAttention(d_model, n_head, dropout = dropout)
        self.norm_attn = nn.LayerNorm(d_model)

        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model*2, kernel_size = 1),
            nn.GLU(dim = 1),
            nn.Conv1d(d_model, d_model, kernel_size = kernel_size, padding = (kernel_size -1)//2, groups = d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size = 1),
            nn.Dropout(dropout)
        )

        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout)
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Feed Forward
        x = x + 0.5*self.ffn1(x)

        # MultiHead Attention
        temp = x
        x = self.norm_attn(x)
        x = self.attn(x, x, x)
        x = x + temp

        # ConV
        temp = x
        x = self.norm_conv(x)
        x = x.transpose(1,2)
        x = self.conv(x)
        x = x.transpose(1,2)
        x = x + temp

        # Feed Forward
        x = x + 0.5*self.ffn2(x)

        return self.final_norm(x)

# STT
class ConformerSTT(nn.Module):
    def __init__(self, n_mels, num_classes, d_model, n_head, num_blocks):
        super().__init__()
        self.Sampling = Conv2d(n_mels, d_model)
        self.blocks = nn.ModuleList([Conformer(d_model, n_head) for _ in range(num_blocks)])
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.Sampling(x)
        for block in self.blocks:
            x = block(x)
        return self.fc(x)