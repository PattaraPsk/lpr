@echo off
setlocal enabledelayedexpansion

title Thai LPR Dataset Relabeling Studio
color 0B

echo =====================================================================
echo           Thai License Plate Dataset Relabeling Studio
echo     [Detection: YOLO Bounding Boxes] ^| [OCR: 77-Province Transcription]
echo =====================================================================
echo.

cd /d "%~dp0"

:: Detect Python executable (prefer pipeline .venv if available)
set "PY_BIN=python"
if exist "%~dp0pipeline\.venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0pipeline\.venv\Scripts\python.exe"
    echo [OK] Using dedicated PyTorch/Ultralytics virtual environment.
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0.venv\Scripts\python.exe"
    echo [OK] Using local virtual environment.
) else (
    echo [INFO] Using system Python.
)

echo [START] Starting Relabeling Studio Web GUI on port 8501...
echo.

cd /d "%~dp0tools\relabel_gui"
"%PY_BIN%" server.py 8501

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Server exited with error code %ERRORLEVEL%.
    pause
)
