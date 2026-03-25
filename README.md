# 🚗 Sonni AIoT Car - Voice Command System

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)

**Sonni AIoT Car** là một hệ thống lõi (Core System) giúp điều khiển các thiết bị xe IoT (ESP32, Arduino, Raspberry Pi) bằng giọng nói Tiếng Việt thông qua trình duyệt Web và Điện thoại di động.

Thay vì sử dụng các mô hình Deep Learning cồng kềnh, hệ thống áp dụng triết lý **Pragmatic Engineering**: Sử dụng Rule-based NLP kết hợp Google STT để đạt tốc độ phản hồi tính bằng mili-giây với độ chính xác 100%.

## ✨ Tính năng nổi bật
- **Ghi âm trực tiếp:** Thu âm giọng nói trực tiếp từ trình duyệt PC hoặc Mobile mà không cần tải file.
- **Xử lý linh hoạt:** Hỗ trợ nhận diện các câu lệnh tự nhiên (VD: *"Đi thẳng rồi rẽ trái"*).
- **Silent Fail UX:** Trải nghiệm người dùng mượt mà, tự động hủy lệnh khi môi trường ồn hoặc ghi âm lỗi mà không văng cảnh báo khó chịu.
- **Mobile Ready:** Tích hợp sẵn kiến trúc chạy qua đường hầm Ngrok (HTTPS), cho phép mang điện thoại ra ngoài đường dùng 4G để điều khiển xe ở nhà.

## 🛠️ Công nghệ sử dụng
- **Backend:** Python, FastAPI, Uvicorn
- **Audio Processing:** `SpeechRecognition` (Google STT API), `pydub` (FFmpeg)
- **Frontend:** HTML5, TailwindCSS, Vanilla JS, MediaRecorder API
- **Networking:** Ngrok (HTTPS Tunnel)

---

## ⚙️ Hướng dẫn cài đặt (Installation)

### 1. Yêu cầu hệ thống (Prerequisites)
- Python 3.8 trở lên.
- **FFmpeg:** Cực kỳ quan trọng để xử lý âm thanh.
  - *Windows:* Tải và thêm FFmpeg vào biến môi trường (System PATH).
  - *Mac/Linux:* `brew install ffmpeg` hoặc `sudo apt install ffmpeg`.

### 2. Thiết lập môi trường
Clone dự án và mở Terminal tại thư mục gốc:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
# (Windows)
source venv/Scripts/activate
# (Mac/Linux)
source venv/bin/activate

# Cài đặt thư viện
pip install fastapi uvicorn python-multipart SpeechRecognition pydub
