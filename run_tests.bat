@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   ISO and Global Standards Compliance Test Suite
echo   (ISO/IEC/IEEE 29119 Quality Assurance Verification)
echo ============================================================

REM Find Python in local .venv or PATH
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    echo [OK] Using virtual environment: %~dp0.venv
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=python"
        echo [OK] Using system python
    ) else (
        echo [ERROR] Python not found in .venv or system PATH!
        exit /b 1
    )
)

echo.
echo Running automated ISO test discovery...
echo.

"%PYTHON_EXE%" -m unittest discover -s tests -p "test_*.py" -v

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo [PASSED] All ISO and Enterprise standard compliance tests passed!
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo [FAILED] Some compliance tests failed. Check log output above.
    echo ============================================================
    exit /b 1
)

endlocal
