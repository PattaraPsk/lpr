@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo   CAM01 License Plate Detection & Tracking Runner
echo ============================================================

REM Check Python environment
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH!
    pause
    exit /b 1
)

REM Run detection on CAM01 current folder with auto-tracking
python tools\detect_cam_videos.py --input-dir "dataset\Raw Video Record\CAM01\2026\10\19" %*

echo.
echo [DONE] Processing finished.
