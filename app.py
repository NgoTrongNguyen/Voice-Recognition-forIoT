import os
import shutil
import random
import speech_recognition as sr
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. CORE DNA: MODEL & TOKENIZER CLASSES (GIỮ NGUYÊN)
# ==========================================
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

class CommandTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text):
        return [self.word2idx.get(w, 3) for w in text.lower().split()]

# ==========================================
# 2. UTILS: LOAD MODEL & INFERENCE (GIỮ NGUYÊN)
# ==========================================
def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    src_tok = CommandTokenizer()
    src_tok.word2idx = checkpoint['src_word2idx']
    src_tok.idx2word = {i: w for w, i in src_tok.word2idx.items()}
    
    trg_tok = CommandTokenizer()
    trg_tok.word2idx = checkpoint['trg_word2idx']
    trg_tok.idx2word = {i: w for w, i in trg_tok.word2idx.items()}
    
    enc = Encoder(len(src_tok.word2idx), checkpoint['emb_dim'], checkpoint['hid_dim'], checkpoint['layers'], 0.1)
    attn = Attention(checkpoint['hid_dim'])
    dec = Decoder(len(trg_tok.word2idx), checkpoint['emb_dim'], checkpoint['hid_dim'], checkpoint['layers'], 0.1, attn)
    
    loaded_model = Seq2Seq(enc, dec, device).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    return loaded_model, src_tok, trg_tok

def predict(model, sentence, src_tok, trg_tok, device, max_token=10):
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
            
    return " ".join([trg_tok.idx2word[i] for i in trg_indices if i not in [0,1,2]])

# ==========================================
# 3. FASTAPI APP SETUP (SINGLETON MODEL)
# ==========================================
app = FastAPI(title="Voice to Vehicle Command API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép mọi giao diện gọi tới
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "epoch500.ckpt"
print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
MODEL, SRC_TOK, TRG_TOK = load_model(MODEL_PATH, DEVICE)
RECOGNIZER = sr.Recognizer()

@app.post("/api/v1/command")
async def process_voice_command(audio_file: UploadFile = File(...)):
    temp_original_path = f"temp_raw_{audio_file.filename}"
    temp_wav_path = f"temp_processed.wav"
    
    try:
        # 1. Lưu file nhận được (có thể là mp3, m4a, ogg...)
        with open(temp_original_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # 2. Convert mọi thứ sang WAV thô (PCM) để AI hiểu
        print(f"Đang xử lý file: {audio_file.filename}...")
        audio = AudioSegment.from_file(temp_original_path)
        audio.export(temp_wav_path, format="wav")

        # 3. Đọc file WAV và chuyển thành STT
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = RECOGNIZER.record(source)
            
        recognized_text = RECOGNIZER.recognize_google(audio_data, language="vi-VN")
        print(f"Nhận diện được: {recognized_text}")

        # 4. Cho qua NLP model lấy mã lệnh
        vehicle_command = predict(MODEL, recognized_text, SRC_TOK, TRG_TOK, DEVICE)

        return JSONResponse(content={
            "status": "success",
            "recognized_text": recognized_text,
            "vehicle_command": vehicle_command
        })

    except sr.UnknownValueError:
        return JSONResponse(content={"status": "error", "message": "Không nhận diện được giọng nói trong file."}, status_code=400)
    except sr.RequestError:
        return JSONResponse(content={"status": "error", "message": "Lỗi kết nối đến Google STT API."}, status_code=502)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": f"Lỗi hệ thống: {str(e)}"}, status_code=500)
    finally:
        # 5. CỰC KỲ QUAN TRỌNG: Dọn rác
        if os.path.exists(temp_original_path):
            os.remove(temp_original_path)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)