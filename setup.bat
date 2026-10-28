@echo off
REM ImageBench - one-time environment setup (Windows)
REM Creates .venv and installs requirements.txt
setlocal
cd /d "%~dp0"

set "PYCMD="
for %%V in (3.13 3.12 3.11) do (
    if not defined PYCMD (
        py -%%V -c "import sys" >nul 2>&1 && set "PYCMD=py -%%V"
    )
)
if not defined PYCMD (
    python -c "import sys" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
    echo [ERROR] Python 3.11+ was not found. Install it from https://www.python.org/downloads/
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment with: %PYCMD%
    %PYCMD% -m venv .venv || exit /b 1
)

echo [setup] Installing requirements...
".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1

echo.
echo [setup] Done. Start the dashboard with:  run.bat
endlocal
