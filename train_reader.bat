@echo off
title Train Thai Plate Reader OCR (Edge Thai Plate Reader OCR)
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

echo ============================================================
echo   Train Thai Plate Reader OCR (Edge Thai Plate Reader OCR)
echo ============================================================
set "DATA_DIR=dataset\ready\202610261140,dataset\dataset-cctv-synthetic\plates,dataset\dataset-mobile-synthetic\plates"

echo Datasets  : %DATA_DIR%
echo Accelerator: CUDA (RTX 4090 Mixed Precision AMP)
echo.

%PYTHON_EXE% tools\train_models.py --task reader --data "%DATA_DIR%" --batch 64 --epochs 30 %*
if errorlevel 1 pause
