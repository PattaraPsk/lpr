@echo off
setlocal enabledelayedexpansion

title Selen-LPR Inference Service Runner
color 0B

cd /d "%~dp0"

set "PY_BIN=python"
if exist "..\pipeline\.venv\Scripts\python.exe" set "PY_BIN=..\pipeline\.venv\Scripts\python.exe"

echo ==============================================================================
echo   SELEN-LPR: Thai License Plate Detection & Vehicle Classification Service
echo ==============================================================================
echo.

if "%~1"=="" (
    echo [INFO] Running test inference on sample: samples\01_private_car\sample.png
    "%PY_BIN%" interfaces\service.py --image samples\01_private_car\sample.png
) else (
    "%PY_BIN%" interfaces\service.py %*
)

pause
