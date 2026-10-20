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
echo   [1] Launch OWOD Studio GUI (Auto-Labeling and YOLO Pruning)
echo   [2] Auto-Run OWOD on all data in 'dataset' folder (GPU Batch)
echo   [3] Launch Thai LPR Dataset Annotator and Reviewer
echo   [4] Launch ISO-Standard RTSP Stream Collector
echo   [5] Run CAM01 Video Detection and Tracking Runner
echo   [6] Run ISO Compliance and Quality Test Suite
echo ========================================================
set "choice=1"
set /p choice="Select tool [1-6] (Default 1): "

if "%choice%"=="6" (
    cd /d "%~dp0.."
    call run_tests.bat
) else if "%choice%"=="5" (
    cd /d "%~dp0.."
    call run_detection.bat
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
