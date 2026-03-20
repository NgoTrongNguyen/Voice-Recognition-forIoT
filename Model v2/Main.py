import speech_recognition as sr
import torch
import torch.nn as nn
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




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, src_tokenizer, trg_tokenizer = load_model("epoch500.ckpt", device)

# Khởi tạo recognizer
r = sr.Recognizer()

# Dùng micro để thu âm
with sr.Microphone() as source:
    print("Nói gì đó...")
    audio = r.listen(source)

# Nhận diện bằng Google
try:
    text = r.recognize_google(audio, language="vi-VN")
    print("Bạn nói:", text)
except sr.UnknownValueError:
    print("Không nhận diện được giọng nói")
except sr.RequestError:
    print("Lỗi kết nối đến dịch vụ Google")


res = predict(model, text, src_tokenizer, trg_tokenizer, device, max_token=10)
print(f"Lệnh: {text:15} => Hành động: {res}")