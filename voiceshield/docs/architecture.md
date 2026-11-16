# VoiceShield Architecture Specification

## 1. System Overview

**VoiceShield** is an enterprise audio deepfake detection platform designed to protect banking call centers and authentication pipelines from AI voice cloning. It wraps the **VoiceGuard** AASIST model in an extensible, modular architecture that cleanly isolates audio ingestion, deep learning inference, explainability diagnostics, and real-time streaming protocols.

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

## 2. Core Architectural Components

### A. Frontend Dashboard (`voiceshield/frontend`)
- **Technology**: React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons, Recharts.
- **Design Philosophy**: High-contrast, dark cybersecurity theme (`#0a0d14` background, glassmorphism, glowing risk borders).
- **Views**:
  - **Overview**: Real-time aggregated metrics (Total Analyzed, Likely Human, Likely AI, Uncertain, Average Latency) and risk distribution chart.
  - **Audio Analysis**: Drag-and-drop file uploader, HTML5 player, interactive waveform visualizer with segment overlays, score gauge, and spectral metrics.
  - **Live Call Monitor**: WebSocket-driven streaming visualizer with ring buffer progress meter, rolling suspicion trajectory, and consecutive-window fraud alerts.
  - **Detection History**: Local audit log of analyzed files with search, state filtering, and JSON export.
  - **Model Architecture**: Deep dive into SincConv filterbank, GAT-S, GAT-T, parameter counts, and benchmark findings.
  - **Settings**: Dynamic decision boundary tuning and hyperparameter controls.

### B. Backend REST & WebSocket Layer (`voiceshield/backend/app/api`)
- **Framework**: FastAPI with asynchronous endpoints and CORS middleware.
- **Endpoints**:
  - `GET /api/health`: Health probe returning model status, device (CPU/CUDA), and torch version.
  - `GET /api/model-info`: Architectural metadata, parameters, input constraints, and limitations.
  - `POST /api/analyze`: Multipart file upload returning structured deepfake classification.
  - `POST /api/analyze/batch`: Multi-file classification with concurrency throttling.
  - `POST /api/waveform`: Fast peak extraction for interactive frontend scrubbing.
  - `GET /api/history`: Recent in-memory audit log.
  - `DELETE /api/history`: Clear local analysis history.
  - `WS /api/stream/analyze`: Real-time streaming WebSocket endpoint for PCM audio chunks.

### C. VoiceGuard Integration Service (`voiceshield/backend/app/services/voiceguard_service.py`)
- Acts as a zero-modification adapter around the original `voiceguard` repository.
- Manages PyTorch device mapping (`auto`, `cuda`, `cpu`), warm-up passes, and model evaluation mode.
- Extracts raw logits `[spoof, bonafide]`, performs Softmax projection, and applies calibrated decision boundaries.
- Slices multi-second audio into overlapping 4.04s windows (64,600 samples) and computes whole-file aggregated scores.

### D. Real-Time Call Protection Abstraction (`voiceshield/backend/app/services/call_protection.py`)
Designed to support future enterprise telephony integrations (SIP/RTP, WebRTC, authorized PBX taps):
1. **`AudioStreamSource`**: Abstract base class for receiving live audio chunks.
2. **`AudioBufferManager`**: Thread-safe ring buffer accumulating PCM bytes with configurable window size (default 6.0s) and overlap retention (default 2.0s).
3. **`VoiceDetectionEngine`**: Protocol decoupling the buffer from the underlying detection model.
4. **`DetectionEventHandler`**: Applies Exponential Moving Average (EMA) temporal smoothing ($s_t = \alpha \cdot r_t + (1-\alpha) \cdot s_{t-1}$) and requires sustained consecutive high-risk windows before triggering alerts.
5. **`CallDetectionSession`**: High-level coordinator managing an active call's lifecycle.

---

## 3. Data Flow

### File Upload Flow
1. User uploads audio (WAV, MP3, FLAC, OGG).
2. `validation.py` validates MIME type, file size ($\le 50\text{ MB}$), and duration ($0.5\text{s} - 180\text{s}$).
3. `audio_service.py` decodes audio to float32 mono array and computes spectral diagnostics.
4. `voiceguard_service.py` slices audio into 64,600-sample windows and executes AASIST forward passes.
5. Softmax logits are converted into deepfake probability scores and mapped to 3-tier classification states.
6. Structured `AnalysisResult` is returned to the frontend and recorded in local audit history.

### Live Streaming Flow
1. Client connects to `ws://localhost:8000/api/stream/analyze`.
2. Client streams 16-bit PCM chunks (base64 encoded).
3. `AudioBufferManager` accumulates samples and emits buffer readiness status.
4. When 6.0 seconds are buffered, `AASIST` analyzes the window.
5. `DetectionEventHandler` updates the smoothed trajectory and checks consecutive alert thresholds.
6. Real-time packet is emitted to the client; trailing 2.0s overlap is retained in the buffer.
