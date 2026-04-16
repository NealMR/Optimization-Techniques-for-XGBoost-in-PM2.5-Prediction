@echo off
echo Starting PM2.5 Sentinel Live Dashboard...
echo Loading environment from .env...

:: Check if streamlit is installed
python -c "import streamlit" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Streamlit is not installed.
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

:: Run the streamlit app
streamlit run app.py

pause
