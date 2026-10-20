@echo off
setlocal enabledelayedexpansion

title Thai LPR - Synthetic License Plate Generator
color 0E

cd /d "%~dp0\synthetic_data"

set "VENV_ACT="
if exist "..\..\pipeline\.venv\Scripts\activate.bat" set "VENV_ACT=..\..\pipeline\.venv\Scripts\activate.bat"
if not defined VENV_ACT if exist "..\.venv\Scripts\activate.bat" set "VENV_ACT=..\.venv\Scripts\activate.bat"

if not defined VENV_ACT (
    echo [ERROR] Virtual environment not found.
    pause
    exit /b 1
)

call "%VENV_ACT%"

echo ==============================================================================
echo   GENERATING SYNTHETIC THAI LICENSE PLATES (All 77 Provinces)
echo ==============================================================================
echo.

python augment_license_plate.py %*

pause
