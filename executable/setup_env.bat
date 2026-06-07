@echo off
title NYC Taxi Project - Setup Environment
echo =====================================================================
echo                NYC Taxi Project Environment Setup
echo =====================================================================
echo.

:: Move to the project root directory
cd /d "%~dp0.."

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your PATH.
    echo Please install Python 3.10 or 3.11 first and try again.
    echo.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if exist venv (
    echo [INFO] Virtual environment 'venv' already exists in the project root.
    echo Re-installing / verifying requirements...
    echo.
) else (
    echo [INFO] Creating virtual environment 'venv' in the project root...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
    echo.
)

:: Activate the virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo.
echo [INFO] Installing packages from requirements.txt...
echo This might take a few minutes depending on your internet connection...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo [SUCCESS] Environment setup complete!
echo You can now run the other batch files in this folder.
echo =====================================================================
echo.
pause
