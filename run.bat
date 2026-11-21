@echo off
cd /d "%~dp0"
python -c "import cv2, PySide6, numpy" 2>nul || (
    echo Installing requirements...
    python -m pip install -r requirements.txt
)
python main.py
if errorlevel 1 pause
