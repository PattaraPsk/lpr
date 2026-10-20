@echo off
setlocal enabledelayedexpansion

title Master Thai LPR Pipeline Launcher
color 0A

cd /d "%~dp0\pipeline"

set "PY_BIN=python"
if exist ".venv\Scripts\python.exe" (
    set "PY_BIN=.venv\Scripts\python.exe"
)

if "%~1"=="" (
    echo Launching Thai LPR Benchmark on sample media...
    "%PY_BIN%" main.py --mode benchmark --source tests/samples/sample.mp4 --max-frames 30
) else (
    "%PY_BIN%" main.py %*
)
