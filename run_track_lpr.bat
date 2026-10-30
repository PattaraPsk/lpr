@echo off
setlocal
chcp 65001 >nul

set VENV_PYTHON=%~dp0.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" set VENV_PYTHON=python

echo ===============================================================================
echo  THAI LPR VIDEO PROCESSOR, TRACKER & OPTIMAL SNAPSHOT EXTRACTOR
echo ===============================================================================

"%VENV_PYTHON%" "tools\process_and_track_lpr.py" %*

echo.
echo [DONE] Processing complete.
pause
