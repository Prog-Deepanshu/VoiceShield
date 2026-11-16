"""
VoiceGuard — Audio Analysis Route
====================================
File-upload endpoint for analyzing pre-recorded audio files.
Supports WAV, FLAC, and raw PCM formats.
"""

import io
import logging
import uuid
from datetime import datetime

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from api.middleware import verify_api_key
from api.schemas import AnalysisResult, SpectralAnalysisResponse
from core.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Analyze audio file for deepfake detection",
    description=(
        "Upload an audio file and receive a deepfake analysis result. "
        "Supports WAV, FLAC, and raw PCM formats. "
        "Audio should be 2-30 seconds of speech at 16kHz mono."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def analyze_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to analyze"),
    call_id: str = Form(
        default=None,
        description="Unique call identifier (auto-generated if not provided)",
    ),
    sample_rate: int = Form(
        default=16000,
        description="Sample rate for raw PCM files (ignored for WAV/FLAC)",
    ),
) -> AnalysisResult:
    """
    Analyze an uploaded audio file for synthetic speech.

    The file is read, preprocessed, and passed through the deepfake
    detection model(s). Returns a risk assessment with score and
    recommended action.
    """
    app = request.app

    # Generate call_id if not provided
    if not call_id:
        call_id = f"UPLOAD-{uuid.uuid4().hex[:8].upper()}"

    # Read the uploaded file
    try:
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # Decode audio
    try:
        audio, sr = _decode_audio(file_content, file.filename, sample_rate)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode audio: {e}. Supported formats: WAV, FLAC, PCM.",
        )

    # Validate duration
    duration_sec = len(audio) / sr
    if duration_sec < 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too short ({duration_sec:.1f}s). Minimum 1 second required.",
        )
    if duration_sec > 60.0:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too long ({duration_sec:.1f}s). Maximum 60 seconds allowed.",
        )

    logger.info(
        f"Analyzing {file.filename}: {duration_sec:.1f}s, {sr}Hz, call_id={call_id}"
    )

    # ── Run AASIST inference ─────────────────────────────────────────
    aasist_result = app.state.aasist_engine.infer(audio, sample_rate=sr)

    # ── Optional Wav2Vec2 ensemble ───────────────────────────────────
    wav2vec2_result = None
    if (
        hasattr(app.state, "wav2vec2_engine")
        and app.state.wav2vec2_engine is not None
        and app.state.wav2vec2_engine.is_loaded
    ):
        wav2vec2_result = app.state.wav2vec2_engine.infer(audio, sample_rate=sr)

    # ── Ensemble scoring ─────────────────────────────────────────────
    ensemble_result = app.state.ensemble_scorer.score(
        aasist_result, wav2vec2_result
    )

    # ── Risk classification ──────────────────────────────────────────
    risk = app.state.risk_classifier.classify(
        ensemble_result["deepfake_score"],
        ensemble_result["confidence"],
    )

    # ── Spectral analysis (diagnostics) ──────────────────────────────
    feature_extractor = FeatureExtractor(sample_rate=sr)
    spectral = feature_extractor.analyze(audio)

    # ── Update metrics ───────────────────────────────────────────────
    _update_metrics(app, ensemble_result, risk)

    # ── Trigger alert if HIGH risk ───────────────────────────────────
    if risk.should_alert and hasattr(app.state, "alert_service"):
        try:
            await app.state.alert_service.send_alert_async(
                call_id=call_id,
                score=ensemble_result["deepfake_score"],
                risk_level=risk.risk_level.value,
            )
        except Exception as e:
            logger.error(f"Alert dispatch failed: {e}")

    # ── Audit log ────────────────────────────────────────────────────
    if hasattr(app.state, "audit_logger"):
        app.state.audit_logger.log_analysis(
            call_id=call_id,
            score=ensemble_result["deepfake_score"],
            risk_level=risk.risk_level.value,
            action=risk.recommended_action.value,
            latency_ms=ensemble_result["total_latency_ms"],
            models_used=ensemble_result["models_used"],
        )

    return AnalysisResult(
        call_id=call_id,
        deepfake_score=ensemble_result["deepfake_score"],
        risk_level=risk.risk_level.value,
        recommended_action=risk.recommended_action.value,
        risk_description=risk.description,
        confidence=ensemble_result["confidence"],
        inference_latency_ms=ensemble_result["total_latency_ms"],
        models_used=ensemble_result["models_used"],
        model_scores=ensemble_result["model_scores"],
        timestamp=datetime.utcnow(),
        spectral_analysis=SpectralAnalysisResponse(**spectral.to_dict()),
    )


def _decode_audio(
    content: bytes, filename: str | None, fallback_sr: int
) -> tuple[np.ndarray, int]:
    """
    Decode audio bytes into numpy array.

    Tries soundfile first (WAV/FLAC/OGG), falls back to raw PCM.
    """
    try:
        # Try soundfile (handles WAV, FLAC, OGG, etc.)
        audio, sr = sf.read(io.BytesIO(content), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        return audio, sr
    except Exception:
        # Assume raw PCM (16-bit signed little-endian)
        audio = np.frombuffer(content, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, fallback_sr


def _update_metrics(app, ensemble_result: dict, risk) -> None:
    """Update application-level inference metrics."""
    latency = ensemble_result["total_latency_ms"]

    app.state.total_analyses = getattr(app.state, "total_analyses", 0) + 1

    if risk.risk_level.value == "HIGH":
        app.state.total_high_risk = getattr(app.state, "total_high_risk", 0) + 1
    elif risk.risk_level.value == "MEDIUM":
        app.state.total_medium_risk = getattr(app.state, "total_medium_risk", 0) + 1

    # Running average latency
    n = app.state.total_analyses
    prev_avg = getattr(app.state, "avg_latency_ms", 0.0)
    app.state.avg_latency_ms = prev_avg + (latency - prev_avg) / n

    app.state.min_latency_ms = min(
        getattr(app.state, "min_latency_ms", float("inf")), latency
    )
    app.state.max_latency_ms = max(
        getattr(app.state, "max_latency_ms", 0.0), latency
    )
