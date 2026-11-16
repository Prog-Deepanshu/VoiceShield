#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  Starting VoiceShield FastAPI Backend (Port 8000)..."
echo "============================================================"

# Resolve project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_DIR}/voiceshield/backend:${PROJECT_DIR}/voiceguard"

# Activate venv if exists
if [ -f "${PROJECT_DIR}/venv/bin/activate" ]; then
    source "${PROJECT_DIR}/venv/bin/activate"
fi

python3 -m uvicorn app.main:app --app-dir "${PROJECT_DIR}/voiceshield/backend" --host 0.0.0.0 --port 8000 --reload
