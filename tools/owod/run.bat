@echo off
title OWOD License Plate Studio - YOLO Pruning
cd /d "%~dp0"
python main.py
if errorlevel 1 pause
