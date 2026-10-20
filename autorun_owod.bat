@echo off
title Auto-Run OWOD Dataset Labeling
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

%PYTHON_EXE% tools/owod/autorun_dataset.py
pause
