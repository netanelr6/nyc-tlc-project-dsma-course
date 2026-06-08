@echo off
title NYC TLC Taxi Fare Predictor Launcher
echo ============================================================
echo      NYC TLC Taxi Fare Predictor -- Streamlit Launcher
echo ============================================================
echo.
echo [1/2] Launching Streamlit web server...
echo.

if exist nyc_taxi_venv\Scripts\python.exe (
    echo [2/2] Using virtual environment nyc_taxi_venv...
    nyc_taxi_venv\Scripts\python.exe -m streamlit run streamlit_app.py
) else if exist .venv\Scripts\python.exe (
    echo [2/2] Using virtual environment .venv...
    .venv\Scripts\python.exe -m streamlit run streamlit_app.py
) else if exist venv\Scripts\python.exe (
    echo [2/2] Using virtual environment venv...
    venv\Scripts\python.exe -m streamlit run streamlit_app.py
) else if exist C:\Users\netan\anaconda3\envs\ny_taxis\python.exe (
    echo [2/2] Using ny_taxis conda environment...
    C:\Users\netan\anaconda3\envs\ny_taxis\python.exe -m streamlit run streamlit_app.py
) else (
    echo [2/2] Using system python...
    python -m streamlit run streamlit_app.py
)

echo.
echo Streamlit server has stopped.
echo.
pause
