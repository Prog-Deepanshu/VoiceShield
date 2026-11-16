<div align="center">

# 🛡️ VoiceGuard

### Real-Time AI Voice Deepfake Detection for Banking Call Centers

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Detect synthetic voices in real-time — before fraud happens.*

</div>

---

## 🎯 What is VoiceGuard?

VoiceGuard is a production-grade deepfake detection system that analyzes **live call audio** and identifies synthetic speech within the **first 5–10 seconds** of a call. It's designed to protect banking call centers from AI-powered voice cloning fraud.

### The Problem
Modern generative AI can clone a person's voice from only a few seconds of audio. Fraudsters use this to bypass voice biometric authentication or impersonate customers during phone calls. Traditional defenses (phone number verification, caller ID) are easily spoofed.

### The Solution
VoiceGuard provides audio-level deepfake detection using state-of-the-art pretrained anti-spoofing models:

- **AASIST** — Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks (primary detector)
- **Wav2Vec2** — Fine-tuned deepfake audio classifier from HuggingFace (optional ensemble verifier)

---

## 🏗️ Architecture

```
Live Call Audio (SIP/RTP/WebSocket)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Audio Stream Buffer (ring buffer, 5-10s)       │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  Preprocessor                                    │
│  ├─ Resample to 16kHz                           │
│  ├─ Normalize [-1, 1]                           │
│  ├─ Voice Activity Detection                     │
│  └─ Spectral-gating noise reduction             │
└─────────────────┬───────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
┌──────────────┐   ┌──────────────────┐
│ AASIST Model │   │ Wav2Vec2 Model   │
│ (85K params) │   │ (optional)       │
│ Primary      │   │ Ensemble         │
└──────┬───────┘   └────────┬─────────┘
       │                    │
       └────────┬───────────┘
                ▼
┌─────────────────────────────────────────────────┐
│  Ensemble Scorer                                 │
│  └─ Weighted average (0.7×AASIST + 0.3×W2V2)   │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  Risk Classifier                                 │
│  ├─ score < 0.3  → 🟢 LOW    → ALLOW           │
│  ├─ 0.3 ≤ s < 0.7 → 🟡 MEDIUM → FLAG_FOR_REVIEW│
│  └─ score ≥ 0.7  → 🔴 HIGH   → BLOCK_AND_ALERT │
└─────────────────┬───────────────────────────────┘
                  │
          ┌───────┼───────┐
          ▼       ▼       ▼
       REST    WebSocket  Webhook
       API     Alert      Callback
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
cd deepfake
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

```bash
python scripts/download_weights.py

# Include Wav2Vec2 for ensemble mode:
python scripts/download_weights.py --include-wav2vec2
```

### 3. Start the Server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 4. Test with an Audio File

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@test_audio.wav" \
  -F "call_id=TEST-001"
```

### 5. View API Docs

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger documentation.

---

## 🐳 Docker Deployment

```bash
# CPU mode
docker-compose up -d

# GPU mode (requires NVIDIA Container Toolkit)
docker-compose --profile gpu up -d

# Check health
curl http://localhost:8000/health
```

---

## 📡 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service information |
| `GET` | `/health` | Health check |
| `GET` | `/health/ready` | Readiness probe (K8s) |
| `GET` | `/health/metrics` | Inference statistics |
| `POST` | `/api/v1/analyze` | Analyze uploaded audio file |

### WebSocket Streaming

Connect to `ws://localhost:8000/api/v1/stream/{call_id}` for real-time analysis.

**Client → Server:**
```json
{"type": "audio_chunk", "data": "<base64 PCM>", "seq": 1}
{"type": "end_stream"}
```

**Server → Client:**
```json
{"type": "buffer_status", "buffered_seconds": 3.2, "target_seconds": 6.0, "is_ready": false}
{"type": "analysis_result", "data": {"deepfake_score": 0.87, "risk_level": "HIGH", ...}}
```

### Example Response

```json
{
  "call_id": "CALL-2026-04-11-001",
  "deepfake_score": 0.87,
  "risk_level": "HIGH",
  "recommended_action": "BLOCK_AND_ALERT",
  "risk_description": "High probability of synthetic speech detected (score: 0.870)...",
  "confidence": 0.92,
  "inference_latency_ms": 34.5,
  "models_used": ["aasist"],
  "model_scores": {"aasist": 0.87},
  "spectral_analysis": {
    "spectral_centroid_hz": 1523.45,
    "pitch_mean_hz": 142.3,
    "pitch_stability": 0.0456,
    "harmonic_ratio": 2.34
  }
}
```

---

## ⚙️ Configuration

Configuration via environment variables or `.env` file:

```bash
cp .env.example .env
# Edit .env with your settings
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VARIANT` | `aasist_l` | `aasist` (297K params) or `aasist_l` (85K params) |
| `ENABLE_ENSEMBLE` | `false` | Enable Wav2Vec2 ensemble |
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `BUFFER_DURATION_SEC` | `6.0` | Audio buffer duration (seconds) |
| `RISK_THRESHOLD_LOW` | `0.3` | Below = LOW risk |
| `RISK_THRESHOLD_HIGH` | `0.7` | At/Above = HIGH risk |
| `WEBHOOK_URL` | | Fraud alert webhook URL |
| `API_KEY` | | API authentication key |

---

## 📊 Performance

Latency benchmarks (P95):

| Configuration | GPU (V100) | GPU (RTX 3060) | CPU (i7-12700) |
|---|---|---|---|
| AASIST-L only | ~25ms | ~40ms | ~160ms |
| AASIST full | ~50ms | ~80ms | ~300ms |
| AASIST-L + preprocessing | ~40ms | ~60ms | ~190ms |
| AASIST-L + Wav2Vec2 ensemble | ~120ms | ~180ms | ~500ms+ |

Run your own benchmark:
```bash
python scripts/benchmark.py --model aasist_l --device auto --iterations 100
```

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/ -v

# Specific test module
pytest tests/test_audio_buffer.py -v
pytest tests/test_inference.py -v
```

---

## 📁 Project Structure

```
deepfake/
├── config/                     # Configuration
│   ├── settings.py             # Pydantic settings
│   ├── aasist.json             # AASIST model config
│   └── aasist_l.json           # AASIST-L model config
├── models/
│   ├── aasist.py               # AASIST architecture
│   └── weights/                # Pretrained weights (.pth)
├── core/
│   ├── audio_buffer.py         # Streaming audio buffer
│   ├── preprocessor.py         # Audio preprocessing
│   ├── feature_extractor.py    # Spectral diagnostics
│   ├── inference_engine.py     # Model inference
│   ├── ensemble.py             # Multi-model scoring
│   └── risk_classifier.py      # Risk level classification
├── api/
│   ├── app.py                  # FastAPI application
│   ├── schemas.py              # Pydantic models
│   ├── middleware.py           # Auth & logging
│   └── routes/
│       ├── health.py           # Health endpoints
│       ├── analyze.py          # File upload analysis
│       └── stream.py           # WebSocket streaming
├── services/
│   ├── alert_service.py        # Fraud alert dispatch
│   └── audit_logger.py         # Compliance logging
├── scripts/
│   ├── download_weights.py     # Download pretrained weights
│   └── benchmark.py            # Latency benchmarking
├── tests/                      # Unit & integration tests
├── Dockerfile                  # Production container
├── docker-compose.yml          # Orchestration
└── requirements.txt            # Dependencies
```

---

## 📄 Models & References

### AASIST
- **Paper:** [Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks](https://arxiv.org/abs/2110.01200) (ICASSP 2022)
- **Repository:** [clovaai/aasist](https://github.com/clovaai/aasist)
- **Trained on:** ASVspoof 2019 LA dataset
- **Performance:** EER 0.83% (AASIST), EER 0.99% (AASIST-L)

### Wav2Vec2 Deepfake Detector
- **Model:** [Hemgg/Deepfake-audio-detection](https://huggingface.co/Hemgg/Deepfake-audio-detection)
- **Base:** facebook/wav2vec2-base
- **Accuracy:** 95.5% on evaluation set

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

AASIST model code is © 2021-present NAVER Corp., used under MIT license.
