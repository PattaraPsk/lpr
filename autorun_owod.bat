@echo off
title Auto-Run OWOD Dataset Labeling
cd /d "%~dp0"
python tools/owod/autorun_dataset.py
pause
