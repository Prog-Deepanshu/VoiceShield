# VoiceGuard Implementation Walkthrough

The VoiceGuard system has been fully implemented according to the designed architecture. It provides a real-time AI voice deepfake detection module tailored for banking call centers.

## What Was Built

The system includes the complete production pipeline spanning from audio ingestion to alert dispatch:

### 1. Model Architectures & Inference Engines
- **AASIST Model** (`models/aasist.py`): The core graph attention network architecture adapted from NAVER Corp's official implementation to support low-latency inference on raw waveforms.
- **Inference Engines** (`core/inference_engine.py`): Dedicated engines for AASIST and an optional Wav2Vec2 ensemble model. Features include:
  - TorchScript tracing for performance.
  - Optional Half-Precision (FP16) on GPUs.
  - CUDA kernel warm-up steps to avoid cold-start delays.

### 2. Core Processing Pipeline
- **Audio Stream Buffer** (`core/audio_buffer.py`): A thread-safe ring buffer that accumulates streaming PCM chunks reliably.
- **Audio Preprocessor** (`core/preprocessor.py`): Performs zero-latency VAD (Voice Activity Detection), spectral-gating noise reduction, resampling to 16kHz, and normalization.
- **Feature Extractor** (`core/feature_extractor.py`): Calculates diagnostic features like pitch stability, spectral flatness, and zero-crossing rates for auditing and explanations.

### 3. Risk & Ensemble Policies
- **Ensemble Scorer** (`core/ensemble.py`): Merges AASIST and Wav2Vec2 predictions into a unified, confidence-weighted score.
- **Risk Classifier** (`core/risk_classifier.py`): Maps raw scores into clear, actionable `RiskLevels` (LOW, MEDIUM, HIGH) configurable via `.env`, which decide whether to allow, review, or block.

### 4. Enterprise API Layer
- **REST Endpoints** (`api/routes/analyze.py`): Sync endpoints for analyzing full uploaded audio files.
- **WebSocket Streaming** (`api/routes/stream.py`): Real-time endpoint handling chunked audio streams, making use of sliding-window inference.
- **System APIs** (`api/routes/health.py`): Readiness and health probes optimized for Kubernetes deployment, tracking running analytics (like average latency).
- **Services** (`services/`): Webhook-driven fraud notification (`alert_service.py`) and JSONL immutable audit logging (`audit_logger.py`).

### 5. Deployment and Validation Tools
- **Scripts**: Built `download_weights.py` to seamlessly acquire pretrained models from HF and clovaai, and `benchmark.py` to evaluate the system latency under load.
- **Docker & Compose**: Fully containerized setup featuring independent GPU and CPU profiles.

## Next Steps

1. **Environment Setup**: Ensure you execute `scripts/download_weights.py` to pull the AI weights from HuggingFace and GitHub.
2. **Virtual Environment**: Run `python -m venv .venv` and source it, then `pip install -r requirements.txt`.
3. **Execution**: Start the server via `uvicorn api.app:app` or use the provided `docker-compose.yml`.
4. Run the benchmark tool (`python scripts/benchmark.py`) to verify local hardware performance against the 200ms latency budget.
