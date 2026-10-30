@echo off
setlocal
title Thai LPR - Plate Augmentor
cd /d "%~dp0"

REM Locate virtual environment python
if exist "..\..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\..\.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" main.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
endlocal
