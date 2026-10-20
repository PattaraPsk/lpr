@echo off
setlocal enabledelayedexpansion

title Thai LPR - Video File Processor
color 0B

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

if not "%~1"=="" (
    python main.py --mode video --source "%~1" --config configs/config.default.yaml
) else (
    python main.py --mode video --source tests/samples/sample.mp4 --config configs/config.default.yaml
)

pause
