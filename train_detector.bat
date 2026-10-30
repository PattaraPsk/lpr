@echo off
title Train Plate Detector ^& Vehicle Classifier (YOLO)
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

echo ============================================================
echo   Train Plate Detector ^& Vehicle Classifier (YOLO)
echo ============================================================
set "DATA_YAML=dataset\labeled\dataset-vehicle-orientation\data.yaml"
if not exist "%DATA_YAML%" set "DATA_YAML=dataset\dataset-vehicle-orientation\data.yaml"

echo Dataset : %DATA_YAML%
echo.

%PYTHON_EXE% tools\train_models.py --task detector --data "%DATA_YAML%" %*
if errorlevel 1 pause
