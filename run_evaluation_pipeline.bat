@echo off
title NYC TLC Taxi Fare Predictor Launcher
echo ============================================================
echo      NYC TLC Taxi Fare Predictor -- Running Act 1, 5, and 6
echo ============================================================
echo.

set PYTHON_CMD=python

if exist nyc_taxi_venv\Scripts\python.exe (
    echo [1/5] Using virtual environment nyc_taxi_venv...
    set PYTHON_CMD=nyc_taxi_venv\Scripts\python.exe
) else if exist .venv\Scripts\python.exe (
    echo [1/5] Using virtual environment .venv...
    set PYTHON_CMD=.venv\Scripts\python.exe
) else if exist venv\Scripts\python.exe (
    echo [1/5] Using virtual environment venv...
    set PYTHON_CMD=venv\Scripts\python.exe
) else if exist C:\Users\netan\anaconda3\envs\ny_taxis\python.exe (
    echo [1/5] Using ny_taxis conda environment...
    set PYTHON_CMD=C:\Users\netan\anaconda3\envs\ny_taxis\python.exe
) else (
    echo [1/5] Using system python...
)

echo.
echo [2/5] Running Act 1 (Data Prep and Feature Engineering)...
%PYTHON_CMD% pipeline.py --act 1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Act 1 failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/5] Running Act 5 (Staging and Testing Hold-out Evaluation)...
%PYTHON_CMD% pipeline.py --act 5
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Act 5 failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [4/5] Running Act 6 (Deployment Preparation and Verification)...
%PYTHON_CMD% pipeline.py --act 6
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Act 6 failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [5/5] Launching Streamlit web application...
call run_streamlit.bat
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Streamlit launcher exited with a non-zero status.
)

echo.
echo Pipeline execution (Act 1, 5, 6, and Streamlit Launch) completed successfully!
echo.
pause
