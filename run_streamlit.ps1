# BBC News RAG System - Streamlit Frontend Launcher
Write-Host "Starting BBC News RAG System - Streamlit Frontend" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install/update dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Run Streamlit app
Write-Host "Starting Streamlit application..." -ForegroundColor Green
Write-Host ""
Write-Host "The app will open in your default web browser." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application." -ForegroundColor Cyan
Write-Host ""

streamlit run streamlit_app.py
