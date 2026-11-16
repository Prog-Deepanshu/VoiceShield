"""
VoiceShield — Pydantic Schemas
==============================
Data transfer models for VoiceShield REST and WebSocket endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClassificationState(str, Enum):
    """The three user-facing classification states."""
    LIKELY_HUMAN = "LIKELY HUMAN"
    LIKELY_AI_GENERATED = "LIKELY AI-GENERATED"
    UNCERTAIN = "UNCERTAIN"


class RiskLevel(str, Enum):
    """Internal model risk level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AudioMetadata(BaseModel):
    """Metadata describing the uploaded audio file."""
    filename: str
    duration_sec: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int


class SegmentAnalysis(BaseModel):
    """Segment-level prediction from sliding window evaluation."""
    window_index: int
    start_sec: float
    end_sec: float
    score: float
    classification: ClassificationState
    spoof_score: float
    bonafide_score: float


class SpectralDiagnostics(BaseModel):
    """Diagnostic acoustic features extracted for explainability."""
    pitch_stability: float = Field(..., description="Normalized pitch variation (lower = more flat/synthetic)")
    pitch_mean_hz: float = Field(..., description="Estimated fundamental frequency (F0)")
    spectral_flatness: float = Field(..., description="Wiener entropy (higher = noisier/vocoder-like)")
    spectral_centroid_hz: float = Field(..., description="Center of mass of spectrum")
    spectral_rolloff_hz: float = Field(..., description="Frequency below which 85% of energy lies")
    harmonic_ratio: float = Field(..., description="Energy ratio of harmonics to residual")
    energy_rms: float = Field(..., description="Root-mean-square amplitude")


class AnalysisResult(BaseModel):
    """Standardized response from audio classification."""
    id: str
    filename: str
    classification: ClassificationState
    model_score: float = Field(..., description="Softmax deepfake probability score [0.0, 1.0]")
    confidence: float = Field(..., description="Confidence metric reflecting decision certainty [0.0, 1.0]")
    raw_logits: Optional[List[float]] = Field(default=None, description="Raw model logits [spoof_logit, bonafide_logit]")
    model_name: str
    model_version: str
    audio_metadata: AudioMetadata
    processing_duration_ms: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    segments: List[SegmentAnalysis] = Field(default_factory=list, description="Per-window predictions across the audio")
    spectral_diagnostics: Optional[SpectralDiagnostics] = None
    limitations_note: str = Field(
        default="Trained on ASVspoof 2019 benchmark. Scores reflect acoustic similarity to synthetic speech artifacts and should not be treated as definitive legal proof of fraud."
    )
    is_uncertain: bool = False
    uncertainty_reason: Optional[str] = None


class BatchAnalysisResult(BaseModel):
    """Response for batch audio file analysis."""
    total_files: int
    human_count: int
    ai_generated_count: int
    uncertain_count: int
    total_duration_sec: float
    total_processing_ms: float
    results: List[AnalysisResult]


class HealthStatus(BaseModel):
    """Response for /api/health probe."""
    status: str
    model_loaded: bool
    model_variant: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    torch_version: str
    uptime_sec: float
    version: str


class ModelInfoResponse(BaseModel):
    """Comprehensive details about the underlying VoiceGuard model."""
    architecture: str
    model_variant: str
    parameter_count: int
    input_requirements: Dict[str, Any]
    weights_path: str
    weights_found: bool
    available_methods: List[str]
    supported_inference_modes: List[str]
    decision_thresholds: Dict[str, float]
    calibration_method: str
    important_limitations: List[str]


class StreamChunkMessage(BaseModel):
    """Incoming WebSocket audio chunk message."""
    type: str = "audio_chunk"
    data: str = Field(..., description="Base64-encoded PCM audio bytes (16-bit signed, 16kHz mono)")
    seq: int = 0


class BufferStatusMessage(BaseModel):
    """Outgoing WebSocket buffer readiness status."""
    type: str = "buffer_status"
    buffered_seconds: float
    target_seconds: float
    is_ready: bool


class StreamResultData(BaseModel):
    """Live analysis packet emitted over WebSocket."""
    call_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    analysis_index: int
    window_score: float
    smoothed_score: float
    classification: ClassificationState
    alert_triggered: bool
    latency_ms: float
    buffered_seconds: float


class StreamAnalysisMessage(BaseModel):
    """WebSocket wrapper message for stream analysis."""
    type: str = "analysis_result"
    data: StreamResultData
