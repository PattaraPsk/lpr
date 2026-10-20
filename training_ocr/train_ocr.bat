@echo off
setlocal enabledelayedexpansion

title Thai LPR - OCR Model Training Launcher
color 0A

cd /d "%~dp0\models"

set "VENV_ACT="
if exist "..\..\pipeline\.venv\Scripts\activate.bat" set "VENV_ACT=..\..\pipeline\.venv\Scripts\activate.bat"
if not defined VENV_ACT if exist "..\.venv\Scripts\activate.bat" set "VENV_ACT=..\.venv\Scripts\activate.bat"

if not defined VENV_ACT (
    echo [ERROR] Virtual environment not found.
    pause
    exit /b 1
)

call "%VENV_ACT%"

echo ==============================================================================
echo   THAI LICENSE PLATE OCR (CRNN + CTC) MODEL TRAINING
echo ==============================================================================
echo   [1] Train CRNN Model (50 Epochs, GPU)
echo   [2] Export Trained CRNN Model to ONNX and Deploy to pipeline/models/
echo   [3] Exit
echo ==============================================================================
echo.

set /p OPTION="Select an option [1-3]: "

if "%OPTION%"=="1" (
    python -m training_ocr.models.train_ocr --epochs 50 --batch 32
) else if "%OPTION%"=="2" (
    python -m training_ocr.models.export_onnx --deploy-to-pipeline
) else (
    echo Exiting.
    exit /b 0
)

pause
