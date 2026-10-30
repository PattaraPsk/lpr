@echo off
title Train SelenLPR (selen-lpr-0.0.1) Unified Multi-Task Model
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

echo ============================================================
echo   Train SelenLPR (selen-lpr-0.0.1) Unified Multi-Task Model
echo   Architecture: Pretrained MobileNetV3 + Multi-Head Attention
echo   Targets     : plate_type, d3, d4, province
echo   Loss        : SelenMultiTaskLoss (Homoscedastic Uncertainty)
echo   Hardware    : NVIDIA RTX 4090 Mixed Precision (AMP)
echo ============================================================
echo.

%PYTHON_EXE% tools\train_selen_lpr.py --epochs 15 --batch-size 64 %*
if errorlevel 1 pause
