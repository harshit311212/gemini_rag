@echo off
title DP-600 RAG Assistant Launcher

echo =======================================================
echo     Starting Data Bear DP-600 RAG Assistant
echo =======================================================

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please install Python and try again.
    pause
    exit /b
)

:: Check for .env file
if not exist .env (
    echo [WARNING] No .env file found! 
    echo Please ensure you copied .env.example to .env and set your API keys.
    echo The application will probably fail if API keys are missing.
    echo.
)

:: Activate Virtual Environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo [INFO] Activated Virtual Environment.
) else (
    echo [WARNING] No .venv found. Running globally.
)

:: Check Dependencies Before Running Pip
echo [INFO] Verifying dependencies...
python -c "import streamlit, chromadb, google.genai, groq, fitz, PIL, dotenv" >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] All dependencies are ready. Skipping slow installation check!
) else (
    echo [INFO] Missing dependencies detected. Downloading necessary packages...
    pip install --default-timeout=1000 -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b
    )
)

:: Run Streamlit App
echo [INFO] Starting Application...
streamlit run app.py

pause
