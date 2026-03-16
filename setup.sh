#!/bin/bash
# NeoVision Setup Script
# Run this ONCE to install all dependencies

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║      NeoVision Smart Glasses — Setup             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Python check
python3 --version || { echo "Python 3.8+ required"; exit 1; }

# Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Upgrade pip
echo "[2/5] Upgrading pip..."
pip install --upgrade pip -q

# Install packages
echo "[3/5] Installing Python packages (this may take a few minutes)..."
pip install flask flask-cors flask-socketio eventlet opencv-python numpy Pillow ultralytics mediapipe pytesseract deep-translator requests

# Tesseract OCR (system level)
echo "[4/5] Checking Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract already installed"
else
    echo ""
    echo "⚠ Tesseract not found. Please install it:"
    echo ""
    echo "  Ubuntu/Debian:  sudo apt install tesseract-ocr"
    echo "  macOS:          brew install tesseract"
    echo "  Windows:        Download from https://github.com/UB-Mannheim/tesseract/wiki"
    echo "                  Then add to PATH"
    echo ""
fi

echo "[5/5] Downloading YOLOv8 model (first run only)..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || echo "YOLO will download on first camera start"

echo ""
echo "✅ Setup complete!"
echo ""
echo "▶ To run NeoVision:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "▶ Then open: http://localhost:5000"
echo ""
