@echo off
title NYC Taxi Project - Run Streamlit App
echo =====================================================================
echo                NYC Taxi Project - Streamlit GUI
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

:: Execute Streamlit
echo [INFO] Starting Streamlit server...
echo.
python -m streamlit run streamlit_app.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Streamlit failed to run.
    pause
    exit /b 1
)
