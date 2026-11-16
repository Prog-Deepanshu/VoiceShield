@echo off
cd /d "%~dp0"
echo ============================================================
echo   Starting VoiceShield FastAPI Backend (Port 8000)...
echo ============================================================
set PYTHONPATH=%CD%\voiceshield\backend;%CD%\voiceguard
.\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir voiceshield/backend --host 0.0.0.0 --port 8000 --reload
pause
