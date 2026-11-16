"""
VoiceShield — API Routes
=========================
REST and WebSocket route definitions for VoiceShield backend.
"""

import base64
import json
import logging
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.models.schemas import (
    AnalysisResult,
    BatchAnalysisResult,
    BufferStatusMessage,
    HealthStatus,
    ModelInfoResponse,
    StreamAnalysisMessage,
    StreamChunkMessage,
    StreamResultData,
)
from app.services.call_protection import CallDetectionSession
from app.services.inference_service import inference_service
from app.utils.validation import (
    validate_audio_duration,
    validate_file_content,
    validate_file_meta,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["VoiceShield"])


# ── Health & Metadata ────────────────────────────────────────────────

@router.get("/health", response_model=HealthStatus, summary="Check service health and model status")
async def get_health():
    """Return backend status, model loading status, and hardware availability."""
    return inference_service.get_health_status()


@router.get("/model-info", response_model=ModelInfoResponse, summary="Retrieve model details and limitations")
async def get_model_info():
    """Return model architecture, weights path, input specs, and critical limitations."""
    return inference_service.get_model_info()


# ── Audio Classification ─────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResult, summary="Analyze an uploaded audio file")
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file to classify (WAV, MP3, FLAC, OGG, PCM)"),
    enable_windowing: bool = Form(default=True, description="Enable multi-window segment evaluation")
):
    """
    Accept an audio file, validate duration and size, execute inference through the
    VoiceGuard AASIST model, and return a comprehensive deepfake risk assessment.
    """
    safe_filename = validate_file_meta(file)
    content = await file.read()
    validate_file_content(content)

    try:
        result = await inference_service.analyze_file(
            file_bytes=content,
            filename=safe_filename,
            enable_windowing=enable_windowing
        )
        validate_audio_duration(result.audio_metadata.duration_sec)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Audio processing error for {safe_filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference failure for {safe_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(e)}")


@router.post("/analyze/batch", response_model=BatchAnalysisResult, summary="Batch analyze multiple audio files")
async def analyze_audio_batch(
    files: List[UploadFile] = File(..., description="Multiple audio files for batch classification")
):
    """
    Accept multiple audio files and run analysis concurrently with concurrency throttling.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch request.")

    files_data = []
    for f in files:
        safe_name = validate_file_meta(f)
        content = await f.read()
        validate_file_content(content)
        files_data.append((safe_name, content))

    try:
        return await inference_service.analyze_batch(files_data)
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch analysis error: {str(e)}")


# ── History Management ───────────────────────────────────────────────

@router.get("/history", response_model=List[AnalysisResult], summary="Get local detection history")
async def get_history(limit: int = 100):
    """Retrieve locally analyzed files in reverse chronological order."""
    return await inference_service.get_history(limit=limit)


@router.delete("/history", summary="Clear local detection history")
async def clear_history():
    """Clear in-memory local detection history."""
    await inference_service.clear_history()
    return {"status": "cleared", "message": "Analysis history cleared successfully."}


# ── Waveform Preview Helper ──────────────────────────────────────────

@router.post("/waveform", summary="Extract waveform peak points for UI rendering")
async def get_waveform_peaks(
    file: UploadFile = File(...),
    points: int = Form(default=120)
):
    """Return normalized peak amplitudes for fast interactive frontend visualization."""
    validate_file_meta(file)
    content = await file.read()
    validate_file_content(content)

    try:
        audio, sr, meta = inference_service.audio_service.decode_audio_bytes(content, file.filename or "audio.wav")
        peaks = inference_service.audio_service.get_waveform_preview(audio, num_points=points)
        return {
            "duration_sec": meta.duration_sec,
            "sample_rate": meta.sample_rate,
            "channels": meta.channels,
            "peaks": peaks
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract waveform: {e}")


# ── WebSocket Streaming Endpoint ──────────────────────────────────────

@router.websocket("/stream/analyze")
async def websocket_stream_analyze(websocket: WebSocket):
    """
    Real-time streaming audio analysis WebSocket endpoint.
    Accepts base64-encoded PCM audio chunks, buffers data into analysis windows,
    evaluates using the VoiceGuard detection engine, applies temporal smoothing,
    and returns continuous stream status and alert triggers.
    """
    await websocket.accept()

    call_id = f"CALL-{websocket.client.host if websocket.client else 'CLIENT'}"
    session = CallDetectionSession(
        call_id=call_id,
        engine=inference_service.voiceguard_service,
        sample_rate=16000,
        window_duration_sec=6.0
    )

    logger.info(f"WebSocket streaming session started: {call_id}")

    try:
        while True:
            text_data = await websocket.receive_text()
            try:
                msg = json.loads(text_data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON message format"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "audio_chunk":
                raw_b64 = msg.get("data", "")
                try:
                    pcm_bytes = base64.b64decode(raw_b64)
                except Exception as b64_err:
                    await websocket.send_json({"type": "error", "message": f"Base64 decode failed: {b64_err}"})
                    continue

                event = session.feed_pcm(pcm_bytes)

                # Send buffer status
                buffer_status = BufferStatusMessage(
                    buffered_seconds=round(session.buffer_manager.buffered_seconds, 2),
                    target_seconds=session.buffer_manager.window_duration_sec,
                    is_ready=session.buffer_manager.is_window_ready()
                )
                await websocket.send_text(buffer_status.model_dump_json())

                # If window was evaluated, send analysis result
                if event is not None:
                    result_payload = StreamAnalysisMessage(
                        data=StreamResultData(
                            call_id=call_id,
                            analysis_index=event["analysis_index"],
                            window_score=event["raw_score"],
                            smoothed_score=event["smoothed_score"],
                            classification=event["classification"],
                            alert_triggered=event["alert_triggered"],
                            latency_ms=event["latency_ms"],
                            buffered_seconds=event["buffered_seconds"]
                        )
                    )
                    await websocket.send_text(result_payload.model_dump_json())

            elif msg_type == "end_stream":
                final_event = session.finalize()
                if final_event is not None:
                    result_payload = StreamAnalysisMessage(
                        data=StreamResultData(
                            call_id=call_id,
                            analysis_index=final_event["analysis_index"],
                            window_score=final_event["raw_score"],
                            smoothed_score=final_event["smoothed_score"],
                            classification=final_event["classification"],
                            alert_triggered=final_event["alert_triggered"],
                            latency_ms=final_event["latency_ms"],
                            buffered_seconds=final_event["buffered_seconds"]
                        )
                    )
                    await websocket.send_text(result_payload.model_dump_json())
                break

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type '{msg_type}'"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {call_id}")
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
