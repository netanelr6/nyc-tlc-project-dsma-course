@echo off
title NYC TLC Taxi Fare Predictor Launcher
echo ============================================================
echo      NYC TLC Taxi Fare Predictor -- Streamlit Launcher
echo ============================================================
echo.
echo [1/2] Launching Streamlit web server...
echo.

C:\Users\netan\anaconda3\envs\ny_taxis\python.exe -m streamlit run streamlit_app.py

echo.
echo Streamlit server has stopped.
echo.
pause
