@echo off
REM ImageBench - launch the dashboard (runs setup automatically on first use)
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run] Virtual environment not found - running setup first...
    call "%~dp0setup.bat" || exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py %*
endlocal
