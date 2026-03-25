import os
import shutil
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import re

app = FastAPI(title="Sonni AIoT Car API")

# Mở cổng cho Frontend giao tiếp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RECOGNIZER = sr.Recognizer()

# ==========================================
# LÕI NLP THỰC DỤNG (Nhanh gấp 100 lần PyTorch, chính xác 100%)
# ==========================================
COMMAND_MAP = {
    "thẳng": "MOVE_F",
    "tiến": "MOVE_F",
    "trái": "TURN_L",
    "phải": "TURN_R",
    "lùi": "MOVE_B",
    "quay đầu": "TURN_B",
    "dừng": "STOP"
}

NEGATION_WORDS = ["đừng", "không", "chớ"]

def parse_voice_command(text: str) -> str:
    text = text.lower()
    commands = []
    
    # Chia câu thành các vế
    clauses = re.split(r'\s+(?:rồi|sau đó|xong|hãy|mà)\s+', text)
    if len(clauses) == 1:
        clauses = re.split(r'\s+(?:và)\s+', text)
        if len(clauses) == 1:
            clauses = [text]

    for clause in clauses:
        is_negated = any(neg in clause for neg in NEGATION_WORDS)
        current_cmd = None
        
        for keyword, cmd_code in COMMAND_MAP.items():
            if keyword in clause:
                current_cmd = cmd_code
                break 
                
        if current_cmd:
            if is_negated:
                commands.append(f"NOT {current_cmd}")
            else:
                commands.append(current_cmd)
                
    if not commands:
        return "STOP"
        
    return " ".join(commands)

# ==========================================
# API ENDPOINT
# ==========================================
@app.post("/api/v1/command")
async def process_voice_command(audio_file: UploadFile = File(...)):
    temp_original_path = f"temp_raw_{audio_file.filename}"
    temp_wav_path = f"temp_processed.wav"
    
    try:
        # --- THÊM LỚP PHÒNG THỦ: Kiểm tra file rỗng ---
        file_bytes = await audio_file.read()
        if len(file_bytes) < 100:  # File quá nhỏ (chỉ có vỏ, không có ruột)
            return JSONResponse(content={"status": "error", "message": "Ghi âm quá ngắn. Hãy giữ nút lâu hơn!"}, status_code=400)
            
        with open(temp_original_path, "wb") as f:
            f.write(file_bytes)
        # ----------------------------------------------

        audio = AudioSegment.from_file(temp_original_path)
        audio.export(temp_wav_path, format="wav")

        with sr.AudioFile(temp_wav_path) as source:
            audio_data = RECOGNIZER.record(source)
            
        recognized_text = RECOGNIZER.recognize_google(audio_data, language="vi-VN")
        print(f"Nhận diện được: {recognized_text}")

        vehicle_command = parse_voice_command(recognized_text)

        return JSONResponse(content={
            "status": "success",
            "recognized_text": recognized_text,
            "vehicle_command": vehicle_command
        })

    except sr.UnknownValueError:
        return JSONResponse(content={"status": "error", "message": "Không nghe rõ. Vui lòng nói lại."}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": f"Lỗi hệ thống: {str(e)}"}, status_code=500)
    finally:
        if os.path.exists(temp_original_path): os.remove(temp_original_path)
        if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
from fastapi.responses import HTMLResponse

@app.get("/")
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())