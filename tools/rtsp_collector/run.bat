@echo off
title ISO RTSP Stream Ingestion Collector
cd /d "%~dp0"
echo ======================================================================
echo    ISO-Standard RTSP Stream Collector (Synapes LPR Data Engine)
echo ======================================================================
echo    [1] Probe RTSP Stream Health (ISO 25012 Diagnostics)
echo    [2] Start Continuous Ingestion (15-min Segments, Lossless MP4)
echo    [3] Start Short Test Ingestion (60-second Test Capture)
echo    [4] Verify Storage Data Quality & SHA-256 Hashes
echo    [5] View Collector Storage Status & Statistics
echo ======================================================================
set /p choice="Select action [1-5] (Default 1): "

if "%choice%"=="2" (
    echo Starting continuous stream ingestion... (Press Ctrl+C to stop)
    python cli.py record
) else if "%choice%"=="3" (
    echo Starting 60-second test capture...
    python cli.py record --duration 60 --segment 60
) else if "%choice%"=="4" (
    python cli.py verify
) else if "%choice%"=="5" (
    python cli.py status
) else (
    python cli.py probe
)

if errorlevel 1 pause
