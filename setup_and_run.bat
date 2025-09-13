@echo off
echo Setting up and running BBC News RAG System...
echo.

echo Step 1: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 2: Installing required packages...
python -m pip install pandas langchain chromadb sentence-transformers

echo Step 3: Running the RAG system...
python main.py

pause
