#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  Starting VoiceShield React Dashboard (Port 5173)..."
echo "============================================================"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}/voiceshield/frontend"

npm run dev
