@echo off
title NYC Taxi Project - Run Full Pipeline
echo =====================================================================
echo              NYC Taxi Project - Full End-to-End Pipeline
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

:: Execute Full Pipeline
echo [INFO] Running full pipeline (Acts 1, 2, 3, and 4)...
echo.
python pipeline.py %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pipeline execution failed.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo [SUCCESS] Full pipeline execution completed successfully!
echo =====================================================================
echo.
pause
