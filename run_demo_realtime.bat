@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
title Thai LPR - Real-Time Demo Launcher

REM Find Python executable
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

REM If arguments were passed directly, forward them to demo_realtime.py
if not "%~1"=="" (
    echo [INFO] Running Thai LPR Real-Time Demo with arguments: %*
    "%PYTHON_EXE%" "%~dp0demo_realtime.py" %*
    goto :end
)

:menu
cls
echo ===============================================================================
echo        THAI LICENSE PLATE RECOGNITION (ALPR/ANPR) - REAL-TIME DEMO
echo ===============================================================================
echo   Edge-optimized Thai ALPR with YOLOv11 + Edge Thai OCR
echo   ISO 8601 / ISO/IEC 25012 / ISO/IEC 27001 / ISO 3166-2:TH Compliant
echo ===============================================================================
echo.
echo   [1] Real-Time Demo on Sample CCTV Video (dataset/raw/19-Oct-2026)
echo   [2] Real-Time Demo on Live RTSP Camera (from .env RTSP_URL)
echo   [3] Real-Time Demo on Live USB Webcam (Camera Index 0)
echo   [4] Real-Time Demo on Inverted Camera (with 180 Flip ON)
echo   [5] Enter Custom Video File Path or RTSP Stream URL
echo   [6] Run Headless Benchmark (50 frames throughput test)
echo   [0] Exit
echo.
echo ===============================================================================
set "choice=1"
set /p choice="Select option [0-6] (Default 1): "
set "choice=%choice: =%"

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :opt_sample
if "%choice%"=="2" goto :opt_rtsp
if "%choice%"=="3" goto :opt_webcam
if "%choice%"=="4" goto :opt_rotate
if "%choice%"=="5" goto :opt_custom
if "%choice%"=="6" goto :opt_bench

echo [WARN] Invalid option selected: %choice%
timeout /t 2 >nul
goto :menu

:opt_sample
echo.
echo [INFO] Launching Real-Time Demo on Sample Video...
"%PYTHON_EXE%" "%~dp0demo_realtime.py"
goto :end

:opt_rtsp
echo.
echo [INFO] Launching Real-Time Demo on Live RTSP Stream from .env...
"%PYTHON_EXE%" "%~dp0demo_realtime.py" --rtsp
goto :end

:opt_webcam
echo.
echo [INFO] Launching Real-Time Demo on USB Webcam Index 0...
"%PYTHON_EXE%" "%~dp0demo_realtime.py" --webcam 0
goto :end

:opt_rotate
echo.
echo [INFO] Launching Real-Time Demo with 180 Flip ON...
"%PYTHON_EXE%" "%~dp0demo_realtime.py" --rotate-180
goto :end

:opt_custom
echo.
set "custom_src="
set /p custom_src="Enter Video Path or RTSP URL: "
if "%custom_src%"=="" (
    echo [WARN] No input entered, returning to menu.
    timeout /t 2 >nul
    goto :menu
)
echo [INFO] Launching Real-Time Demo on custom source...
"%PYTHON_EXE%" "%~dp0demo_realtime.py" --source "%custom_src%"
goto :end

:opt_bench
echo.
echo [INFO] Running Headless Benchmark (50 frames)...
"%PYTHON_EXE%" "%~dp0demo_realtime.py" --headless --max-frames 50
goto :end

:end
echo.
echo [DONE] Session ended.
if "%~1"=="" pause
endlocal
