@echo off
title NYC Taxi Project - Run Act 1 (Data Prep)
echo =====================================================================
echo              NYC Taxi Project - Act 1 (Data Prep)
echo =====================================================================
echo.

:: Move to the project root directory
cd /d "%~dp0.."

:: Check if virtual environment exists
if not exist venv (
    echo [ERROR] Virtual environment 'venv' not found in project root.
    echo Please run 'setup_env.bat' first to create the environment.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Execute Act 1
echo [INFO] Running Act 1 (Data Preparation & Feature Engineering)...
echo.
python pipeline.py --act 1 %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Act 1 execution failed.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo [SUCCESS] Act 1 completed successfully!
echo =====================================================================
echo.
pause
