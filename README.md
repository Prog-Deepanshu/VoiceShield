<div align="center">

# 🛡️ VoiceShield

### Enterprise AI Voice Deepfake Detection & Call Protection Platform
*A robust wrapper and operational application around the VoiceGuard AASIST detection engine by TiElixir.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.3+-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7+-3178c6.svg)](https://www.typescriptlang.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4+-38bdf8.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 🎯 Executive Summary

**VoiceShield** is a production-grade application designed to detect synthetic speech, AI voice cloning, and audio deepfakes in both recorded files and real-time audio streams. It integrates directly with the pretrained **AASIST** (*Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*) model from the **VoiceGuard** project.

Instead of outputting uncalibrated binary flags or treating deepfake probability as criminal intent, VoiceShield provides:
- **Calibrated 3-tier classification**: `LIKELY HUMAN`, `LIKELY AI-GENERATED`, and `UNCERTAIN`.
- **Sliding-window multi-segment evaluation**: Evaluates audio across multiple 4.04-second windows to detect spliced deepfake segments across long recordings.
- **Explainable spectral diagnostics**: Analyzes pitch stability, spectral flatness, and harmonic ratios alongside the deep learning model.
- **Real-time call protection abstraction**: Implements ring buffering and exponential moving average (EMA) temporal smoothing to prevent false-alarm call disruptions.
- **Modern cybersecurity dashboard**: Interactive waveform viewer, audio player, audit history, and empirical benchmark documentation.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Frontend Dashboard (React/TS)               │
│   Overview • Audio Analysis • Live Stream • History • Info  │
└──────────────────────────────┬──────────────────────────────┘
                               │  REST / WebSocket
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
│   /api/health • /api/model-info • /api/analyze • /stream    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              VoiceShield Integration Layer                  │
│   Concurrency Throttle • Audio Normalization • Diagnostics  │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│ VoiceGuard AASIST Engine    │ │ Real-Time Call Protection   │
│ SincConv + ResNet + Dual GAT│ │ AudioBufferManager (Ring)   │
│ Heterogeneous Graph Fusion  │ │ Temporal Smoother (EMA)     │
└──────────────┬──────────────┘ └──────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│            3-Tier Calibrated Risk Classification            │
│      LIKELY HUMAN (<0.30) • UNCERTAIN • LIKELY AI (≥0.70)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python**: 3.10 or 3.11
- **Node.js**: 18+ and npm
- **Git**

### 2. Clone & Setup Pretrained Weights
Pretrained AASIST weights from NAVER Clova AI are downloaded into `voiceguard/models/weights`:
```bash
python voiceguard/scripts/download_weights.py
```
*Downloaded weights: `AASIST-L.pth` (85K params) and `AASIST.pth` (297K params).*

### 3. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r voiceshield/backend/requirements.txt

# Run automated tests
cd voiceshield/backend
pytest -c pytest.ini tests -v
cd ../..

# Start FastAPI server
start_backend.bat
# Or manually:
python -m uvicorn app.main:app --app-dir voiceshield/backend --host 0.0.0.0 --port 8000 --reload
```
API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Frontend Setup
```bash
cd voiceshield/frontend
npm install
npm run dev
```
Dashboard available at: [http://localhost:5173](http://localhost:5173)

### 5. Launch Everything Together (Windows)
```bash
start_all.bat
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health probe returning model status, device, and uptime |
| `GET` | `/api/model-info` | Architectural metadata, parameter counts, and limitations |
| `POST` | `/api/analyze` | Upload audio file (WAV, MP3, FLAC, OGG) for classification |
| `POST` | `/api/analyze/batch` | Upload multiple files for concurrent batch analysis |
| `POST` | `/api/waveform` | Extract normalized waveform peak values for UI scrubbing |
| `GET` | `/api/history` | Retrieve in-memory session analysis history |
| `DELETE` | `/api/history` | Clear session audit history |
| `WS` | `/api/stream/analyze` | Real-time WebSocket streaming endpoint for PCM audio chunks |

### Example Analysis Response (`POST /api/analyze`)
```json
{
  "id": "ANL-3D8B2F9A01",
  "filename": "customer_call.wav",
  "classification": "LIKELY AI-GENERATED",
  "model_score": 0.8245,
  "confidence": 0.649,
  "raw_logits": [2.314, -0.842],
  "model_name": "VoiceGuard-AASIST",
  "model_version": "aasist_l-v1.0",
  "audio_metadata": {
    "filename": "customer_call.wav",
    "duration_sec": 8.42,
    "sample_rate": 16000,
    "channels": 1,
    "format": "wav",
    "file_size_bytes": 269504
  },
  "processing_duration_ms": 312.4,
  "segments": [
    {
      "window_index": 0,
      "start_sec": 0.0,
      "end_sec": 4.04,
      "score": 0.7896,
      "classification": "LIKELY AI-GENERATED",
      "spoof_score": 0.7896,
      "bonafide_score": 0.2104
    },
    {
      "window_index": 1,
      "start_sec": 2.0,
      "end_sec": 6.04,
      "score": 0.8542,
      "classification": "LIKELY AI-GENERATED",
      "spoof_score": 0.8542,
      "bonafide_score": 0.1458
    }
  ],
  "spectral_diagnostics": {
    "pitch_stability": 0.0215,
    "pitch_mean_hz": 142.3,
    "spectral_flatness": 0.003412,
    "spectral_centroid_hz": 1642.5,
    "spectral_rolloff_hz": 3150.0,
    "harmonic_ratio": 1.84,
    "energy_rms": 0.04512
  },
  "limitations_note": "Trained on ASVspoof 2019 benchmark. Scores reflect acoustic similarity to synthetic speech artifacts and should not be treated as definitive legal proof of fraud.",
  "is_uncertain": false
}
```

---

## 📊 Empirical Limitations & Domain Shift

AASIST was trained on the clean **ASVspoof 2019 Logical Access (LA)** dataset. During out-of-distribution evaluation against modern benchmarks (recorded in `voiceguard/benchmark_dataset_report.txt`), the original repository noted that high-fidelity audio or lossy codecs (AMR, MP3) can trigger false positives on real human speech:

1. **Sample Rate Discrepancy**: Resampling modern 44.1kHz or 48kHz audio to 16kHz creates steep roll-offs that mimic neural vocoder filtering.
2. **Lossy Compression**: Codec quantization artifacts strip subtle harmonics, elevating synthetic scores.
3. **Background Noise**: Room echo, environmental noise, or background music alter spectral flatness.

### How VoiceShield Solves This:
- **3-Tier Risk Model**: Inconclusive scores ($0.30 \le s < 0.70$) are labeled `UNCERTAIN` instead of falsely accusing callers.
- **Multi-Window Evaluation**: Rather than judging a 30s call on the first 4s, VoiceShield evaluates overlapping windows across the entire call timeline.
- **Spectral Diagnostics**: Compares pitch stability and vocoder noise floor alongside raw model logits for verifiable explainability.

---

## 🐳 Docker Deployment

```bash
docker-compose up -d --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:5173

---

## 🧪 Testing

The backend includes a comprehensive automated test suite verifying model loading, audio decoding, sliding-window chunking, buffer management, temporal smoothing, and API endpoints:

```bash
cd voiceshield/backend
pytest -c pytest.ini tests -v
```
**Result**: 26 tests passed (100% success rate).

---

## 📜 License & Acknowledgments

- **VoiceShield**: MIT License.
- **VoiceGuard Engine**: Authored by TiElixir under MIT License.
- **AASIST Model**: NAVER Corp. (Jung et al., ICASSP 2022) under MIT License.
