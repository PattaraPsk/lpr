@echo off
title SelenLPR v2 (selen-lpr-0.0.1) 500-Epoch Training Runner
REM ===========================================================================
REM SelenLPR v2 (selen-lpr-0.0.1) Continued Training Runner to 500 Epochs
REM Complies with ISO/IEC 42001:2023 & Thailand DLT Domain Specifications
REM Hardware: NVIDIA RTX 4090 Mixed Precision (AMP FP16)
REM ===========================================================================

cd /d "%~dp0"
call .venv\Scripts\activate.bat

python tools\train_selen_lpr_v2.py --epochs 500 --batch-size 64 --lr 0.0003 --loss-mode clamped_kendall --resume models/reader/selen-lpr-0.0.1/best_reader.pth --device cuda:0 %*

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Training completed successfully.
pause
