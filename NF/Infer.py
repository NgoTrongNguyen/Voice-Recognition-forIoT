import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "<SEP>": 4, 
                     "MOVE_F": 5, "MOVE_B": 6, "TURN_L": 7, "TURN_R": 8, "STOP": 9, "TURN_AROUND": 10}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def fit(self, sentences):
        for line in sentences:
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
        # Chỉ lấy các token hành động (id >= 5) hoặc theo nhu cầu của bạn
        return " ".join([self.itos[i] for i in ids if i not in [0, 1, 2, 3, 4]])

# ==========================================
# 3. HÀM LOAD VÀ INFER
# ==========================================
def load_and_infer(checkpoint_path, corpus_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dữ liệu để rebuild Vocab (Tokenizer)
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines() if line.strip()]
    
    tokenizer = RobotTokenizer()
    tokenizer.fit(data)

    # 2. Khởi tạo mô hình (Thông số phải khớp lúc train)
    model = GPT(vocab_size=tokenizer.vocab_size, d_model=128, n_layers=4, n_heads=4, dropout=0.1, block_size=64)
    
    # 3. Load trọng số
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"✅ Đã load model từ {checkpoint_path}")
    print("-" * 30)

    while True:
        user_input = input("Nhập lệnh (ví dụ: 'đi thẳng' - nhấn 'q' để thoát): ")
        if user_input.lower() == 'q': break

        # Chuẩn hóa prompt: thêm <SEP> và đảm bảo có dấu cách
        prompt = user_input.strip() + " <SEP>"
        
        # Encode (không thêm SOS/EOS tự động vì ta tự quản lý cấu trúc prompt)
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special=False)]).to(device)

        # Generate
        # Dùng top_k=1 (Greedy Search) để robot phản hồi chính xác nhất, không 'sáng tạo'
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=9, 
            temperature=0.15, 
            top_k=1, 
            eos_token_id=2
        )

        # Lấy phần ID mới được sinh ra (bỏ phần prompt cũ)
        new_tokens = output_ids[0][input_ids.size(1):].tolist()
        result = tokenizer.decode(new_tokens)

        print(f"🤖 Hành động: {result if result else '[Không nhận diện được]'}")
        print("-" * 30)

if __name__ == "__main__":
    # Thay đổi đường dẫn file của bạn ở đây
    CKPT_FILE = "epoch100.pt" 
    CORPUS_FILE = "corpus.txt"
    
    load_and_infer(CKPT_FILE, CORPUS_FILE)