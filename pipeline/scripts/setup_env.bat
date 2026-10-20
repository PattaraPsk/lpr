@echo off
setlocal enabledelayedexpansion

title Thai LPR Pipeline - 1-Click Environment Setup (Python 3.10.x)
color 0A

echo ==============================================================================
echo   THAI LPR PRODUCTION PIPELINE - ENVIRONMENT SETUP
echo   Target Runtime: Python 3.10.x 64-bit / NVIDIA CUDA 12.4 or CPU
echo ==============================================================================
echo.

cd /d "%~dp0\.."

:: 1. Detect and Enforce Python 3.10
set "PYTHON_310="

py -3.10 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
if !errorlevel! equ 0 set "PYTHON_310=py -3.10"

if not defined PYTHON_310 (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>nul
    if !errorlevel! equ 0 set "PYTHON_310=python"
)

if not defined PYTHON_310 if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PYTHON_310=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PYTHON_310 if exist "C:\Program Files\Python310\python.exe" set "PYTHON_310=C:\Program Files\Python310\python.exe"
if not defined PYTHON_310 if exist "C:\Python310\python.exe" set "PYTHON_310=C:\Python310\python.exe"
if not defined PYTHON_310 if exist "D:\Python310\python.exe" set "PYTHON_310=D:\Python310\python.exe"
if not defined PYTHON_310 if exist "%ProgramFiles%\Python310\python.exe" set "PYTHON_310=%ProgramFiles%\Python310\python.exe"

if not defined PYTHON_310 (
    echo [WARNING] Dedicated Python 3.10 not found in standard paths.
    echo Defaulting to 'python' from system PATH...
    set "PYTHON_310=python"
)

echo [*] Detected Python interpreter: !PYTHON_310!
!PYTHON_310! --version
echo.

:: 2. Create or verify root virtual environment (.venv)
if not exist ".venv" (
    echo [*] Creating isolated virtual environment in .\.venv ...
    !PYTHON_310! -m venv .venv
    if !errorlevel! neq 0 (
        color 0C
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [+] Virtual environment created successfully.
) else (
    echo [*] Existing virtual environment found (.venv).
)

call ".venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

:: 3. Upgrade build tools
echo.
echo [*] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel --quiet

:: 4. Install PyTorch with Hardware Auto-Detection
echo.
echo [*] Detecting hardware acceleration...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [+] NVIDIA GPU detected! Installing PyTorch with CUDA 12.4 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
) else (
    echo [!] No NVIDIA GPU detected. Installing PyTorch CPU build...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

:: 5. Install Master Dependencies
echo.
echo [*] Installing production dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Dependency installation encountered errors.
    pause
    exit /b 1
)

:: 6. Run Self-Test Import Check
echo.
echo [*] Running system self-check and diagnostic...
python scripts\check_environment.py

echo.
echo ==============================================================================
echo [SUCCESS] Thai LPR Production Environment is fully configured and ready!
echo ==============================================================================
pause
