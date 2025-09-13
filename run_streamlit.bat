@echo off
echo Starting BBC News RAG System - Streamlit Frontend
echo ================================================

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run Streamlit app
echo Starting Streamlit application...
echo.
echo The app will open in your default web browser.
echo Press Ctrl+C to stop the application.
echo.

streamlit run streamlit_app.py

pause
