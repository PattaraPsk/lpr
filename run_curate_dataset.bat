@echo off
setlocal enabledelayedexpansion

title Thai LPR Dataset Curation & Labeling Pipeline
color 0B

cd /d "%~dp0"

set "PY_BIN=python"
if exist "pipeline\.venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0pipeline\.venv\Scripts\python.exe"
)

set "GDRIVE_DEFAULT=G:\My Drive\Synapes\LPR"
set "GDRIVE_FALLBACK=G:\.shortcut-targets-by-id\1IDYjuwxlMjypeuFXydazmyoFcMqnjbGS\LPR"
set "OUTPUT_DEFAULT=dataset\curated_lpr_dataset"

echo ======================================================================
echo   Automated Thai LPR Dataset Curation & Cryptographic Deduplication
echo ======================================================================

if "%~1"=="" (
    if exist "!GDRIVE_DEFAULT!" (
        echo [OK] Detected Synapes Google Drive mount at: !GDRIVE_DEFAULT!
        set "SRC_DIR=!GDRIVE_DEFAULT!"
    ) else if exist "!GDRIVE_FALLBACK!" (
        echo [OK] Detected Google Drive shortcut mount at: !GDRIVE_FALLBACK!
        set "SRC_DIR=!GDRIVE_FALLBACK!"
    ) else (
        echo [!] Google Drive mount not found at default location.
        set "SRC_DIR=dataset\storage\verified"
        echo     Using fallback local source: !SRC_DIR!
    )
    echo Running pipeline in COPY mode (Dry-run test recommended first)...
    "%PY_BIN%" dataset\curate_lpr_dataset.py --source "!SRC_DIR!" --output "!OUTPUT_DEFAULT!" --mode copy
) else (
    "%PY_BIN%" dataset\curate_lpr_dataset.py %*
)

pause
