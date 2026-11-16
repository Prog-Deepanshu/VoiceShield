#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

echo "============================================================"
echo "  VoiceShield — AI Voice Deepfake Detection Launcher"
echo "============================================================"

# 1. Setup Python venv if missing
if [ ! -f "venv/bin/python" ]; then
    echo "[SETUP 1/3] Creating virtual environment..."
    python3 -m venv venv
    echo "[SETUP 2/3] Installing backend packages..."
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r voiceshield/backend/requirements.txt
fi

# 2. Check model weights
if [ ! -f "voiceguard/models/weights/AASIST.pth" ]; then
    echo "[SETUP 3/3] Downloading AASIST model weights..."
    ./venv/bin/python voiceguard/scripts/download_weights.py
fi

# 3. Setup Frontend if missing
if [ ! -d "voiceshield/frontend/node_modules" ]; then
    echo "[SETUP] Installing frontend packages..."
    (cd voiceshield/frontend && npm install)
fi

echo "All dependencies ready! Starting VoiceShield..."

# Start backend in background
bash "${PROJECT_DIR}/start_backend.sh" &
BACKEND_PID=$!

# Trap exit to kill backend
trap "kill ${BACKEND_PID} 2>/dev/null || true" EXIT

# Wait for backend to initialize
sleep 3

# Start frontend
bash "${PROJECT_DIR}/start_frontend.sh"
