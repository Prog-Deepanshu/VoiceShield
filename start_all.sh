#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  Launching VoiceShield (Backend + Frontend)..."
echo "============================================================"

# Start backend in background
bash "${PROJECT_DIR}/start_backend.sh" &
BACKEND_PID=$!

# Trap exit to kill backend
trap "kill ${BACKEND_PID} 2>/dev/null || true" EXIT

# Wait for backend to initialize
sleep 3

# Start frontend
bash "${PROJECT_DIR}/start_frontend.sh"
