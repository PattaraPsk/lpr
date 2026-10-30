@echo off
setlocal
chcp 65001 >nul

set VENV_PYTHON=%~dp0.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" set VENV_PYTHON=python

echo ===============================================================================
echo  THAI LPR SYNTHETIC MOBILE DATASET GENERATOR (CALIBRATED V5)
echo ===============================================================================

"%VENV_PYTHON%" "tools\generate_mobile_cctv_synthetic_dataset.py" %*

echo.
echo [DONE] Execution complete.
pause
