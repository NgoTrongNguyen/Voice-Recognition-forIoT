import torch
import torch.nn as nn
import torch.optim as optim
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(hid_dim + emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        # Context Vector
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2))
        return prediction.squeeze(1), hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if random.random() < (teacher_forcing_ratio) else top1
        return outputs



class CommandTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        
    def fit(self, sentences):
        for sent in sentences:
            for word in sent.lower().replace("|", "").split():
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word[len(self.word2idx)-1] = word

    def encode(self, text):
        return [self.word2idx.get(w, 3) for w in text.lower().split()]



def train_loop(model, data, src_tok, trg_tok, device):
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    current_teacher_ratio = 0.7  
    min_ratio = 0.1        
    decay_rate = 0.005       
    
    print(f"Device: {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        current_teacher_ratio = max(min_ratio, 1.0 - (epoch * decay_rate))
        for src_sent, trg_sent in data:
            src_ids = [1] + src_tok.encode(src_sent) + [2]
            trg_ids = [1] + trg_tok.encode(trg_sent) + [2]
            
            src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
            trg_tensor = torch.LongTensor(trg_ids).unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            output = model(src_tensor, trg_tensor, current_teacher_ratio)
            
            # Tính loss bỏ qua token <SOS>
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg_target = trg_tensor[:, 1:].reshape(-1)
            
            loss = criterion(output, trg_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03} | Loss: {epoch_loss/len(data):.4f}")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'src_word2idx': src_tok.word2idx,
                'trg_word2idx': trg_tok.word2idx,
                'hid_dim': hid_dim,
                'emb_dim': emb_dim,
                'layers': layers
            }
            torch.save(checkpoint, f"epoch{epoch+1}.ckpt")

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    
    # Khôi phục Tokenizer
    src_tok = CommandTokenizer()
    src_tok.word2idx = checkpoint['src_word2idx']
    src_tok.idx2word = {i: w for w, i in src_tok.word2idx.items()}
    
    trg_tok = CommandTokenizer()
    trg_tok.word2idx = checkpoint['trg_word2idx']
    trg_tok.idx2word = {i: w for w, i in trg_tok.word2idx.items()}
    
    # Khởi tạo khung Model với các thông số cũ
    enc = Encoder(len(src_tok.word2idx), checkpoint['emb_dim'], checkpoint['hid_dim'], checkpoint['layers'], 0.1)
    attn = Attention(checkpoint['hid_dim'])
    dec = Decoder(len(trg_tok.word2idx), checkpoint['emb_dim'], checkpoint['hid_dim'], checkpoint['layers'], 0.1, attn)
    
    loaded_model = Seq2Seq(enc, dec, device).to(device)
    
    # Nạp trọng số
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval() # Chuyển sang chế độ dự đoán
    
    return loaded_model, src_tok, trg_tok


def predict(model, sentence, src_tok, trg_tok, device, max_token):
    model.eval()
    with torch.no_grad():
        tokens = [1] + src_tok.encode(sentence) + [2]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
        encoder_outputs, hidden = model.encoder(src_tensor)
        
        trg_indices = [1]
        for _ in range(max_token):
            trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
            pred_token = output.argmax(1).item()
            if pred_token == 2: break
            trg_indices.append(pred_token)
            
    return [trg_tok.idx2word[i] for i in trg_indices if i not in [0,1,2]]


raw_data = [
    # --- NHÓM 1: LỆNH ĐƠN CƠ BẢN ---
    ("đi thẳng", "MOVE_F"),
    ("tiến lên", "MOVE_F"),
    ("rẽ trái", "TURN_L"),
    ("quẹo trái", "TURN_L"),
    ("rẽ phải", "TURN_R"),
    ("quẹo phải", "TURN_R"),
    ("lùi lại", "MOVE_B"),
    ("quay đầu", "TURN_B"),
    ("dừng lại", "STOP"),

    # --- NHÓM 2: LỆNH GHÉP (SEQUENCE) ---
    ("quay đầu rồi đi thẳng", "TURN_B MOVE_F"),
    ("quẹo trái rồi tiến thẳng", "TURN_L MOVE_F"),
    ("đi thẳng rồi rẽ phải", "MOVE_F TURN_R"),
    ("lùi lại sau đó dừng lại", "MOVE_B STOP"),
    ("rẽ phải rồi quay đầu", "TURN_R TURN_B"),

    # --- NHÓM 3: LOGIC PHỦ ĐỊNH (NEGATION) ---
    ("đừng rẽ trái hãy đi thẳng", "NOT TURN_L MOVE_F"),
    ("đừng rẽ phải mà đi thẳng", "NOT TURN_R MOVE_F"),
    ("không rẽ trái hãy quẹo phải", "NOT TURN_L TURN_R"),
    ("đừng dừng lại hãy tiến lên", "NOT STOP MOVE_F"),
    ("đừng quay đầu mà lùi lại", "NOT TURN_B MOVE_B"),
    
    # --- NHÓM 4: CÂU LỆNH TỰ NHIÊN (VARIATIONS) ---
    ("hãy đi thẳng đi", "MOVE_F"),
    ("làm ơn quẹo trái", "TURN_L"),
    ("xong rồi dừng lại", "STOP")
]

# Cấu hình
epochs = 500
lr = 0.001
dropout = 0.1
emb_dim = 64
hid_dim = 256
layers = 2
max_token = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Init Tokenizers
src_tokenizer = CommandTokenizer()
trg_tokenizer = CommandTokenizer()
src_tokenizer.fit([x[0] for x in raw_data])
trg_tokenizer.fit([x[1] for x in raw_data])
             
# Init Model
enc = Encoder(len(src_tokenizer.word2idx), emb_dim, hid_dim, layers, dropout)
attn = Attention(hid_dim)
dec = Decoder(len(trg_tokenizer.word2idx), emb_dim, hid_dim, layers, dropout, attn)
model = Seq2Seq(enc, dec, device).to(device)

# Training
train_loop(model, raw_data, src_tokenizer, trg_tokenizer, device)

# 2. Thử nghiệm
print("\n--- TEST MODEL ---")
test_cmds = ["quay đầu rồi đi thẳng", "quẹo trái rồi tiến thẳng", "quay đầu", "lùi lại", "đừng rẽ phải mà đi thẳng"]
for cmd in test_cmds:
    res = predict(model, cmd, src_tokenizer, trg_tokenizer, device, max_token)
    print(f"Lệnh: {cmd:15} => Hành động: {res}")