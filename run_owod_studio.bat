@echo off
title OWOD License Plate Studio - YOLO Pruning
cd /d "%~dp0\tools\owod"

set "PYTHON_EXE=python"
if exist "..\..\.venv\Scripts\python.exe" set "PYTHON_EXE=..\..\.venv\Scripts\python.exe"

%PYTHON_EXE% main.py
if errorlevel 1 pause
