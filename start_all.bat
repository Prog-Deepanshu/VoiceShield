@echo off
setlocal

echo ============================================================
echo   VoiceShield - AI Voice Deepfake Detection Launcher
echo ============================================================
echo.

cd /d "%~dp0"

REM 1. Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on your system!
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM 2. Check Node / npm
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js or npm was not found on your system!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM 3. Setup Virtual Environment if missing
if not exist "venv\Scripts\python.exe" (
    echo [SETUP 1/3] Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SETUP 2/3] Installing Python dependencies - please wait...
    .\venv\Scripts\python.exe -m pip install --upgrade pip
    .\venv\Scripts\pip.exe install -r voiceshield\backend\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install backend dependencies.
        pause
        exit /b 1
    )
)

REM 4. Check model weights
if not exist "voiceguard\models\weights\AASIST.pth" (
    echo [SETUP 3/3] Downloading official pretrained AASIST AI model weights...
    .\venv\Scripts\python.exe voiceguard\scripts\download_weights.py
)

REM 5. Setup Frontend if missing
if not exist "voiceshield\frontend\node_modules" (
    echo [SETUP] Installing frontend dashboard dependencies...
    pushd voiceshield\frontend
    call npm install
    popd
)

echo.
echo ============================================================
echo   All dependencies ready! Launching VoiceShield...
echo ============================================================
echo Backend:   http://localhost:8000 (API Docs: http://localhost:8000/docs)
echo Frontend:  http://localhost:5173 (Web Dashboard)
echo ============================================================
echo.

start "VoiceShield Backend (FastAPI)" cmd /k "start_backend.bat"
ping 127.0.0.1 -n 4 >nul
start "VoiceShield Frontend (Vite)" cmd /k "start_frontend.bat"

ping 127.0.0.1 -n 3 >nul
start http://localhost:5173
