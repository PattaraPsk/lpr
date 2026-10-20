@echo off
setlocal enabledelayedexpansion

title Thai LPR - License Plate Detector Training Launcher
color 0C

cd /d "%~dp0"

set "VENV_ACT="
if exist "..\pipeline\.venv\Scripts\activate.bat" set "VENV_ACT=..\pipeline\.venv\Scripts\activate.bat"
if not defined VENV_ACT if exist ".venv\Scripts\activate.bat" set "VENV_ACT=.venv\Scripts\activate.bat"

if not defined VENV_ACT (
    echo [ERROR] Virtual environment not found.
    pause
    exit /b 1
)

call "%VENV_ACT%"

echo ==============================================================================
echo   THAI LICENSE PLATE DETECTION - MODEL TRAINING LAUNCHER
echo ==============================================================================
echo   [1] Prepare/Unpack Datasets from datasets/
echo   [2] Train YOLOv8s Plate Detector (100 Epochs, GPU)
echo   [3] Train YOLOv11n Lightweight Plate Detector (100 Epochs, GPU)
echo   [4] Export Best Model to ONNX and Deploy to pipeline/models/
echo   [5] Evaluate Model on Validation Set
echo   [6] Exit
echo ==============================================================================
echo.

set /p OPTION="Select an action [1-6]: "

if "%OPTION%"=="1" (
    python prepare_dataset.py --extract-all
) else if "%OPTION%"=="2" (
    python train.py --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640 --device 0 --name thai_plate_yolov8s --export-onnx
) else if "%OPTION%"=="3" (
    python train.py --model yolo11n.pt --epochs 100 --batch 32 --imgsz 640 --device 0 --name thai_plate_yolo11n --export-onnx
) else if "%OPTION%"=="4" (
    python export_onnx.py --deploy-to-pipeline
) else if "%OPTION%"=="5" (
    python evaluate.py
) else (
    echo Exiting.
    exit /b 0
)

pause
