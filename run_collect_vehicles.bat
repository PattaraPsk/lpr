@echo off
setlocal
chcp 65001 >nul

set VENV_PYTHON=%~dp0.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" set VENV_PYTHON=python

echo ===============================================================================
echo  THAI LPR FULL-LENGTH MULTI-VIDEO VEHICLE & LICENSE PLATE COLLECTOR
echo ===============================================================================

"%VENV_PYTHON%" "tools\collect_vehicles_and_plates.py" %*

echo.
echo [DONE] Processing complete.
pause
