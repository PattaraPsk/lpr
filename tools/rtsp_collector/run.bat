@echo off
setlocal enabledelayedexpansion
title ISO RTSP Stream Ingestion Collector
cd /d "%~dp0"

REM Find Python in local .venv or system PATH
if exist "%~dp0..\..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo ======================================================================
echo    ISO-Standard RTSP Stream Collector (Synapes LPR Data Engine)
echo ======================================================================
echo    [1] Probe RTSP Stream Health (ISO 25012 Diagnostics)
echo    [2] Start Ingestion Until Camera Offline (15-min Segments)
echo    [3] Start Short Test Ingestion (60-second Test Capture)
echo    [4] Verify Storage Data Quality & SHA-256 Hashes
echo    [5] View Collector Storage Status & Statistics
echo ======================================================================
set /p choice="Select action [1-5] (Default 1): "

if "%choice%"=="2" (
    echo Starting stream ingestion until camera offline...
    "%PYTHON_EXE%" cli.py record --until-offline
) else if "%choice%"=="3" (
    echo Starting 60-second test capture...
    "%PYTHON_EXE%" cli.py record --duration 60 --segment 60
) else if "%choice%"=="4" (
    "%PYTHON_EXE%" cli.py verify
) else if "%choice%"=="5" (
    "%PYTHON_EXE%" cli.py status
) else (
    "%PYTHON_EXE%" cli.py probe
)

if errorlevel 1 pause
endlocal
