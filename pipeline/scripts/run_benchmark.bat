@echo off
setlocal enabledelayedexpansion

title Thai LPR - Edge Performance Benchmark Runner
color 0E

cd /d "%~dp0\.."

set "VENV_ACT="
if exist ".venv\Scripts\activate.bat" set "VENV_ACT=.venv\Scripts\activate.bat"
if not defined VENV_ACT if exist "zero_shot_license_plate_detection\.venv\Scripts\activate.bat" set "VENV_ACT=zero_shot_license_plate_detection\.venv\Scripts\activate.bat"
if not defined VENV_ACT if exist "zero_shot_license_plate_detection\venv\Scripts\activate.bat" set "VENV_ACT=zero_shot_license_plate_detection\venv\Scripts\activate.bat"

if not defined VENV_ACT (
    echo [ERROR] Virtual environment not found. Please run scripts\setup_env.bat first.
    pause
    exit /b 1
)

call "%VENV_ACT%"

echo ==============================================================================
echo   RUNNING THAI LPR EDGE BENCHMARK (Throughput, Latency, & Accuracy)
echo ==============================================================================
python main.py --mode benchmark --config configs/config.default.yaml %*

pause
