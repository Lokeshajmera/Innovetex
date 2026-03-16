@echo off
echo.
echo ╔══════════════════════════════════════════════════╗
echo ║      NeoVision Smart Glasses - Setup (Windows)  ║
echo ╚══════════════════════════════════════════════════╝
echo.

python --version || (echo Python 3.8+ required && pause && exit)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/4] Upgrading pip...
pip install --upgrade pip -q

echo [3/4] Installing packages...
pip install flask flask-cors flask-socketio eventlet opencv-python numpy Pillow ultralytics mediapipe pytesseract deep-translator requests

echo [4/4] Downloading YOLOv8 model...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>nul || echo YOLO will download on first use

echo.
echo Setup complete!
echo.
echo NOTE: Install Tesseract OCR from:
echo https://github.com/UB-Mannheim/tesseract/wiki
echo Add it to your PATH after installing.
echo.
echo To run: 
echo   venv\Scripts\activate
echo   python app.py
echo.
echo Then open: http://localhost:5000
pause
