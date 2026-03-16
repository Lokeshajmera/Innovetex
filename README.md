# 👓 NeoVision Smart Glasses OS

**Real-time computer vision system for smart glasses**  
YOLO v8 · OpenCV · MediaPipe · Tesseract OCR · Flask + SocketIO

---

## What It Does

| Module | Technology | Description |
|--------|-----------|-------------|
| 🤟 Sign Language | MediaPipe + geometry classifier | Detects hand gestures live, converts to text + speaks aloud |
| 🌐 Translation | Deep Translator (Google) | Converts any foreign language → Hindi or Marathi in real-time |
| 📷 OCR | Tesseract + OpenCV preprocessing | Reads letters/text from captured photos, translates them |
| 🧭 Navigation | Google Maps API | AR route overlay sent to the lens display |
| 🔍 YOLO Detection | YOLOv8n | Detects 80+ objects in camera feed in real-time |

---

## Architecture

```
Browser (Dashboard + Lens Display)
        ↕  WebSocket (Socket.IO)
Flask Server (app.py)
        ↕
┌──────────────────────────────────┐
│  OpenCV   → frame capture/stream │
│  YOLO v8  → object detection     │
│  MediaPipe → hand landmarks      │
│  Tesseract → OCR                 │
│  Deep Translator → Hindi/Marathi │
└──────────────────────────────────┘
```

---

## Quick Start

### Step 1 — Install Tesseract OCR (system level)

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**  
Download installer from https://github.com/UB-Mannheim/tesseract/wiki  
After install, add `C:\Program Files\Tesseract-OCR` to your PATH.

---

### Step 2 — Run setup script

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```
Double-click setup_windows.bat
```

**Or manually:**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### Step 3 — Start the server

```bash
source venv/bin/activate        # Windows: venv\Scripts\activate
python app.py
```

Open browser: **http://localhost:5000**

---

## How To Use

### 🤟 Sign Language Tab
1. Click **START CAMERA** in the center
2. Show hand signs to the camera — MediaPipe detects landmarks in real-time
3. Recognized gesture appears in the Sign tab + on the **Lens Display** (right panel)
4. TTS reads the gesture text aloud
5. Translation to Hindi/Marathi shown automatically
6. No camera? Click **SIMULATE GESTURE** to demo all signs

**Supported Signs:** Hello, Stop, Peace, I Love You, Thumbs Up/Down, OK, Pointing, Call Me, Thank You, Wave

---

### 🌐 Translation Tab
1. Select source language (or Auto-detect)
2. Select target: **Hindi** or **Marathi**
3. Type text OR click **MIC INPUT** to speak
4. Click **TRANSLATE** — result appears in large text + on lens
5. TTS reads the translation in the correct Indian language

---

### 📷 OCR Tab — Letter Recognition
1. **Option A:** Click the preview box to **upload an image** (photo of text/letter)
2. **Option B:** Start camera and click **CAPTURE FRAME** to grab the current view
3. Click **RUN OCR** — Tesseract analyzes the image
4. Recognized text shown with confidence score
5. Automatically translated to chosen language
6. Result sent to the Lens Display

**OCR Pipeline:**
- Grayscale conversion
- Noise removal (fastNlMeansDenoising)
- CLAHE contrast enhancement
- Sharpening kernel
- Otsu thresholding
- 2x upscaling for accuracy

---

### 🧭 Navigation Tab
1. Enter starting point (default: Nashik, Maharashtra)
2. Enter destination
3. Choose travel mode
4. Click **OPEN MAPS** → Google Maps opens in new tab with full directions
5. Click **TO LENS** → route info appears on the Lens Display

---

## Lens Display (Right Panel)

The right panel simulates what the smart glasses wearer sees:
- **Cyan card** — Sign language detection
- **Purple card** — Translation result
- **Pink card** — Navigation route
- **Purple card** — OCR text recognition

All cards animate in real-time as the user interacts with the system.

---

## Troubleshooting

**Camera not working?**
- Make sure no other app is using the camera
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in app.py

**MediaPipe not found?**
```bash
pip install mediapipe
```

**Tesseract OCR returns empty?**
- Ensure Tesseract is installed AND in PATH
- Test: `tesseract --version` in terminal
- For Windows: Set `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'` in app.py

**Translation offline?**
- The system uses Google Translate via `deep-translator`
- Needs internet connection for online translation
- Falls back to built-in Hindi/Marathi dictionary when offline

**YOLO model not downloading?**
- First run requires internet to download `yolov8n.pt` (~6MB)
- The model is cached after first download

---

## File Structure

```
neovision/
├── app.py                 ← Main Flask + SocketIO server
├── requirements.txt       ← All Python dependencies
├── setup.sh               ← Linux/Mac setup
├── setup_windows.bat      ← Windows setup
├── README.md              ← This file
└── templates/
    └── index.html         ← Full UI (Dashboard + Lens Display)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 3.0, Flask-SocketIO |
| Computer Vision | OpenCV 4.9, YOLOv8n (Ultralytics) |
| Hand Detection | MediaPipe 0.10 |
| OCR | Tesseract 5.x + pytesseract |
| Translation | deep-translator (Google Translate API) |
| Real-time | WebSocket via Socket.IO |
| Frontend | Vanilla HTML/CSS/JS, Socket.IO client |
| Fonts | Orbitron, Share Tech Mono, Rajdhani |

---

*Built for NeoVision Smart Glasses Project*
