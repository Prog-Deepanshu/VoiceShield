"""
VoiceGuard — WebSocket Streaming Route
========================================
Real-time streaming endpoint for live call audio analysis.

Protocol:
    Client → Server:
        {"type": "audio_chunk", "data": "<base64 PCM>", "seq": 1}
        {"type": "end_stream"}

    Server → Client:
        {"type": "buffer_status", "buffered_seconds": 3.2, "target_seconds": 6.0, "is_ready": false}
        {"type": "analysis_result", "data": <AnalysisResult>}
        {"type": "error", "message": "...", "code": 400}
"""

import base64
import json
import logging
import uuid
from datetime import datetime

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.schemas import (
    AnalysisResult,
    BufferStatusMessage,
    ErrorMessage,
    SpectralAnalysisResponse,
    StreamAnalysisMessage,
)
from config.settings import settings
from core.audio_buffer import AudioStreamBuffer
from core.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Streaming"])


@router.websocket("/api/v1/stream/{call_id}")
async def stream_analysis(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint for real-time streaming audio analysis.

    Flow:
    1. Client connects with a call_id
    2. Client sends base64-encoded PCM audio chunks
    3. Server buffers audio until target duration is reached
    4. Server analyzes buffered audio and sends results
    5. Server can perform continuous analysis with sliding windows
    """
    await websocket.accept()

    # Initialize buffer for this connection
    buffer = AudioStreamBuffer(
        sample_rate=settings.sample_rate,
        target_duration_sec=settings.buffer_duration_sec,
    )

    app = websocket.app
    analysis_count = 0

    logger.info(f"WebSocket connected: call_id={call_id}")

    try:
        while True:
            # Receive message
            try:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await _send_error(websocket, "Invalid JSON message", 400)
                continue
            except WebSocketDisconnect:
                break

            msg_type = message.get("type", "")

            if msg_type == "audio_chunk":
                # Decode base64 audio data
                try:
                    audio_bytes = base64.b64decode(message.get("data", ""))
                except Exception as e:
                    await _send_error(
                        websocket, f"Invalid base64 audio data: {e}", 400
                    )
                    continue

                if len(audio_bytes) == 0:
                    continue

                # Append to buffer
                buffer.append(audio_bytes)

                # Send buffer status update
                status = BufferStatusMessage(
                    buffered_seconds=round(buffer.buffered_seconds, 2),
                    target_seconds=settings.buffer_duration_sec,
                    is_ready=buffer.is_ready(),
                )
                await websocket.send_text(status.model_dump_json())

                # Analyze when buffer is ready
                if buffer.is_ready():
                    analysis_count += 1
                    result = await _run_analysis(
                        app, buffer, call_id, analysis_count
                    )

                    # Send analysis result
                    result_msg = StreamAnalysisMessage(data=result)
                    await websocket.send_text(result_msg.model_dump_json())

                    # Reset buffer for sliding window analysis
                    # Keep the last 2 seconds for context overlap
                    overlap_samples = int(settings.sample_rate * 2.0)
                    audio = buffer.get_audio()
                    buffer.reset()
                    if len(audio) > overlap_samples:
                        buffer.append_array(audio[-overlap_samples:])

                    logger.info(
                        f"[{call_id}] Analysis #{analysis_count}: "
                        f"score={result.deepfake_score:.3f}, "
                        f"risk={result.risk_level}, "
                        f"latency={result.inference_latency_ms:.1f}ms"
                    )

            elif msg_type == "end_stream":
                # Analyze any remaining audio in the buffer
                if buffer.buffered_samples > settings.sample_rate:  # >1s
                    analysis_count += 1
                    result = await _run_analysis(
                        app, buffer, call_id, analysis_count
                    )
                    result_msg = StreamAnalysisMessage(data=result)
                    await websocket.send_text(result_msg.model_dump_json())

                logger.info(
                    f"Stream ended: call_id={call_id}, "
                    f"total_analyses={analysis_count}"
                )
                break

            else:
                await _send_error(
                    websocket,
                    f"Unknown message type: '{msg_type}'. "
                    "Expected 'audio_chunk' or 'end_stream'.",
                    400,
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: call_id={call_id}")
    except Exception as e:
        logger.error(f"WebSocket error for call_id={call_id}: {e}")
        try:
            await _send_error(websocket, f"Internal error: {str(e)}", 500)
        except Exception:
            pass
    finally:
        buffer.reset()


async def _run_analysis(
    app, buffer: AudioStreamBuffer, call_id: str, analysis_count: int
) -> AnalysisResult:
    """Run deepfake analysis on buffered audio."""
    # Get audio from buffer
    audio = buffer.get_audio(max_samples=settings.nb_samp * 2)

    # AASIST inference
    aasist_result = app.state.aasist_engine.infer(
        audio, sample_rate=settings.sample_rate
    )

    # Optional Wav2Vec2 ensemble
    wav2vec2_result = None
    if (
        hasattr(app.state, "wav2vec2_engine")
        and app.state.wav2vec2_engine is not None
        and app.state.wav2vec2_engine.is_loaded
    ):
        wav2vec2_result = app.state.wav2vec2_engine.infer(
            audio, sample_rate=settings.sample_rate
        )

    # Ensemble scoring
    ensemble_result = app.state.ensemble_scorer.score(
        aasist_result, wav2vec2_result
    )

    # Risk classification
    risk = app.state.risk_classifier.classify(
        ensemble_result["deepfake_score"],
        ensemble_result["confidence"],
    )

    # Spectral analysis (lightweight for streaming)
    feature_extractor = FeatureExtractor(sample_rate=settings.sample_rate)
    spectral = feature_extractor.analyze(audio)

    # Update metrics
    app.state.total_analyses = getattr(app.state, "total_analyses", 0) + 1
    n = app.state.total_analyses
    prev_avg = getattr(app.state, "avg_latency_ms", 0.0)
    latency = ensemble_result["total_latency_ms"]
    app.state.avg_latency_ms = prev_avg + (latency - prev_avg) / n

    # Trigger alert if HIGH risk
    if risk.should_alert and hasattr(app.state, "alert_service"):
        try:
            await app.state.alert_service.send_alert_async(
                call_id=call_id,
                score=ensemble_result["deepfake_score"],
                risk_level=risk.risk_level.value,
            )
        except Exception as e:
            logger.error(f"Alert dispatch failed: {e}")

    # Audit log
    if hasattr(app.state, "audit_logger"):
        app.state.audit_logger.log_analysis(
            call_id=call_id,
            score=ensemble_result["deepfake_score"],
            risk_level=risk.risk_level.value,
            action=risk.recommended_action.value,
            latency_ms=latency,
            models_used=ensemble_result["models_used"],
        )

    return AnalysisResult(
        call_id=f"{call_id}-{analysis_count:03d}",
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


async def _send_error(websocket: WebSocket, message: str, code: int) -> None:
    """Send an error message over WebSocket."""
    error = ErrorMessage(message=message, code=code)
    await websocket.send_text(error.model_dump_json())
