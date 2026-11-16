@echo off
echo ============================================================
echo   Launching VoiceShield (Backend + Frontend)...
echo ============================================================
start "VoiceShield Backend (FastAPI)" cmd /k "start_backend.bat"
timeout /t 3 /nobreak >nul
start "VoiceShield Frontend (Vite)" cmd /k "start_frontend.bat"
echo.
echo VoiceShield services launched!
echo Backend:  http://localhost:8000 (Docs: http://localhost:8000/docs)
echo Frontend: http://localhost:5173
echo ============================================================
