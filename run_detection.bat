@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo   CAM01 License Plate Detection & Tracking Runner
echo ============================================================

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

%PYTHON_EXE% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found!
    pause
    exit /b 1
)

REM Run detection on CAM01 current folder with auto-tracking
set "CAM01_DIR=dataset\raw\Raw Video Record\CAM01\2026\10\19"
if not exist "%CAM01_DIR%" set "CAM01_DIR=dataset\Raw Video Record\CAM01\2026\10\19"
%PYTHON_EXE% tools\detect_cam_videos.py --input-dir "%CAM01_DIR%" %*

echo.
echo [DONE] Processing finished.
