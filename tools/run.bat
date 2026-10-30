@echo off
setlocal enabledelayedexpansion
title LPR Toolset Suite
cd /d "%~dp0"

REM Find Python executable
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo ========================================================
echo   Thai LPR and OWOD License Plate Studio
echo ========================================================
echo   [0] Launch Real-Time Thai LPR Demo (Webcam / RTSP / Video)
echo   [1] Launch OWOD Studio GUI (Auto-Labeling and YOLO Pruning)
echo   [2] Auto-Run OWOD on all data in 'dataset' folder (GPU Batch)
echo   [3] Launch Thai LPR Dataset Annotator and Reviewer
echo   [4] Launch ISO-Standard RTSP Stream Collector
echo   [5] Run CAM01 Video Detection and Tracking Runner
echo   [6] Run ISO Compliance and Quality Test Suite
echo   [7] Train Plate Detector ^& Vehicle Classifier (YOLO)
echo   [8] Train Thai Plate Reader OCR (VisionAttentionOCR)
echo   [9] Run Batch LPR Video Tracking ^& Snapshot Extractor
echo   [10] Launch Plate Augmentor GUI (Synthetic Plates ^& Augmentations)
echo ========================================================
set "choice=0"
set /p choice="Select tool [0-10] (Default 0): "

if "%choice%"=="10" (
    cd /d "%~dp0PlateAugmentor"
    "%PYTHON_EXE%" main.py
) else if "%choice%"=="0" (
    "%PYTHON_EXE%" "%~dp0..\cli.py" demo-realtime
) else if "%choice%"=="9" (
    "%PYTHON_EXE%" "%~dp0..\cli.py" track-lpr
) else if "%choice%"=="8" (
    "%PYTHON_EXE%" "%~dp0..\cli.py" train-reader
) else if "%choice%"=="7" (
    "%PYTHON_EXE%" "%~dp0..\cli.py" train-detector
) else if "%choice%"=="6" (
    "%PYTHON_EXE%" "%~dp0..\cli.py" test
) else if "%choice%"=="5" (
    "%PYTHON_EXE%" "%~dp0..\cli.py" detect
) else if "%choice%"=="4" (
    cd /d "%~dp0rtsp_collector"
    call run.bat
) else if "%choice%"=="3" (
    cd /d "%~dp0DataManipulator"
    "%PYTHON_EXE%" main.py
) else if "%choice%"=="2" (
    cd /d "%~dp0.."
    "%PYTHON_EXE%" "%~dp0owod\autorun_dataset.py"
) else (
    cd /d "%~dp0owod"
    "%PYTHON_EXE%" main.py
)
if errorlevel 1 pause
endlocal
