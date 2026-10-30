@echo off
title Train Edge Thai Plate Reader OCR (EdgeAttentionOCR selen-lpr-0.0.1)
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

echo ============================================================
echo   Train Edge Thai Plate Reader OCR (EdgeAttentionOCR selen-lpr-0.0.1)
echo   Architecture: Pretrained MobileNetV3 + Transformer Attention
echo   Parameters  : ~3.8M (14.6 MB) - 75%% smaller than legacy baseline
echo   Hardware    : NVIDIA RTX 4090 Mixed Precision (AMP)
echo ============================================================
echo.

%PYTHON_EXE% tools\train_edge_attention_ocr.py --epochs 15 --batch-size 64 %*
if errorlevel 1 pause
