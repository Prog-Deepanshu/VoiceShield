"""
VoiceGuard — Pydantic Schemas
===============================
Request/response models for the VoiceGuard API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Metadata for file-based audio analysis."""
    call_id: str = Field(
        ...,
        description="Unique call identifier from the banking system",
        examples=["CALL-2026-04-11-001"],
    )
    audio_format: str = Field(
        default="wav",
        description="Audio format: wav, flac, pcm, mp3",
    )
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz",
    )
    caller_id: Optional[str] = Field(
        default=None,
        description="Caller phone number or identifier (for correlation)",
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent handling the call",
    )


class StreamingChunk(BaseModel):
    """WebSocket audio chunk message."""
    type: str = Field(
        default="audio_chunk",
        description="Message type: 'audio_chunk' or 'end_stream'",
    )
    data: str = Field(
        default="",
        description="Base64-encoded PCM audio data",
    )
    seq: int = Field(
        default=0,
        description="Sequence number for ordering",
    )


# ── Response Models ──────────────────────────────────────────────────────

class SpectralAnalysisResponse(BaseModel):
    """Spectral diagnostic features for audit logging."""
    spectral_centroid_hz: float = 0.0
    spectral_bandwidth_hz: float = 0.0
    spectral_rolloff_hz: float = 0.0
    spectral_flatness: float = 0.0
    zero_crossing_rate: float = 0.0
    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0
    pitch_stability: float = 0.0
    harmonic_ratio: float = 0.0
    energy_rms: float = 0.0


class AnalysisResult(BaseModel):
    """Complete deepfake analysis result."""
    call_id: str = Field(
        ...,
        description="Unique call identifier",
    )
    deepfake_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Deepfake probability (0.0=human, 1.0=synthetic)",
    )
    risk_level: str = Field(
        ...,
        description="Risk classification: LOW, MEDIUM, or HIGH",
    )
    recommended_action: str = Field(
        ...,
        description="Recommended action: ALLOW, FLAG_FOR_REVIEW, or BLOCK_AND_ALERT",
    )
    risk_description: str = Field(
        ...,
        description="Human-readable risk assessment description",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the prediction",
    )
    inference_latency_ms: float = Field(
        ...,
        description="Total inference time in milliseconds",
    )
    models_used: list[str] = Field(
        ...,
        description="List of models used for analysis",
    )
    model_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Individual model scores",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Analysis timestamp (UTC)",
    )
    spectral_analysis: Optional[SpectralAnalysisResponse] = Field(
        default=None,
        description="Optional spectral diagnostic data",
    )


class BufferStatusMessage(BaseModel):
    """WebSocket buffer status update."""
    type: str = "buffer_status"
    buffered_seconds: float
    target_seconds: float
    is_ready: bool


class StreamAnalysisMessage(BaseModel):
    """WebSocket analysis result message."""
    type: str = "analysis_result"
    data: AnalysisResult


class ErrorMessage(BaseModel):
    """WebSocket error message."""
    type: str = "error"
    message: str
    code: int = 500


class HealthResponse(BaseModel):
    """Service health check response."""
    status: str = Field(
        ...,
        description="Service status: 'healthy' or 'unhealthy'",
    )
    models_loaded: list[str] = Field(
        default_factory=list,
        description="List of loaded model names",
    )
    gpu_available: bool = Field(
        ...,
        description="Whether a CUDA GPU is available",
    )
    device: str = Field(
        ...,
        description="Current inference device",
    )
    avg_latency_ms: float = Field(
        default=0.0,
        description="Average inference latency in milliseconds",
    )
    total_analyses: int = Field(
        default=0,
        description="Total number of analyses performed",
    )
    uptime_seconds: float = Field(
        default=0.0,
        description="Service uptime in seconds",
    )
    version: str = Field(
        default="1.0.0",
        description="VoiceGuard version",
    )
