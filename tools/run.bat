@echo off
title LPR Toolset Suite
cd /d "%~dp0"
echo ========================================================
echo   Thai LPR and OWOD License Plate Studio
echo ========================================================
echo   [1] Launch OWOD Studio GUI (Auto-Labeling and YOLO Pruning)
echo   [2] Auto-Run OWOD on all data in 'dataset' folder (GPU Batch)
echo   [3] Launch Thai LPR Dataset Annotator and Reviewer
echo   [4] Launch ISO-Standard RTSP Stream Collector
echo   [5] Run CAM01 Video Detection & Tracking Runner
echo ========================================================
set /p choice="Select tool [1-5] (Default 1): "
if "%choice%"=="5" (
    cd /d "%~dp0.."
    call run_detection.bat
) else if "%choice%"=="4" (
    cd /d "%~dp0rtsp_collector"
    call run.bat
) else if "%choice%"=="3" (
    cd /d "%~dp0DataManipulator"
    python main.py
) else if "%choice%"=="2" (
    cd /d "%~dp0..\dataset"
    python "%~dp0owod\autorun_dataset.py"
) else (
    cd /d "%~dp0owod"
    python main.py
)
if errorlevel 1 pause
