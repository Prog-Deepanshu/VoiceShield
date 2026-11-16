@echo off
cd /d "%~dp0"
echo ============================================================
echo   Starting VoiceShield React Dashboard (Port 5173)...
echo ============================================================
cd voiceshield\frontend
call npm run dev
pause
