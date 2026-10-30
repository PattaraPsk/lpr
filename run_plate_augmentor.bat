@echo off
setlocal
title Thai LPR - Plate Augmentor

REM Locate virtual environment python
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

cd /d "%~dp0tools\PlateAugmentor"
"%PYTHON_EXE%" main.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
endlocal
