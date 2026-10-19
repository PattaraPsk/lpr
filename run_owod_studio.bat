@echo off
title OWOD License Plate Studio - YOLO Pruning
cd /d "%~dp0\tools\owod"
python main.py
if errorlevel 1 pause
