"""
VoiceShield — Application Configuration
========================================
Centralized configuration management for VoiceShield with environment variable support.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings

# Resolve paths
_BACKEND_ROOT = Path(__file__).resolve().parent.parent

def _find_voiceguard_dir() -> Path:
    candidates = [
        _BACKEND_ROOT.parent.parent / "voiceguard",
        _BACKEND_ROOT.parent / "voiceguard",
        Path.cwd() / "voiceguard",
        Path.cwd().parent / "voiceguard",
    ]
    for c in candidates:
        if c.exists() and (c / "models").exists():
            return c.resolve()
    # Default fallback
    return (_BACKEND_ROOT.parent.parent / "voiceguard").resolve()

_VOICEGUARD_DIR = _find_voiceguard_dir()


class AppSettings(BaseSettings):
    """VoiceShield Application Settings."""

    # Server settings
    app_name: str = "VoiceShield"
    app_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000,*"

    # Model configuration
    model_variant: str = Field(default="aasist_l", description="'aasist_l' (85K) or 'aasist' (297K)")
    device: str = Field(default="auto", description="'auto', 'cpu', or 'cuda'")
    enable_ensemble: bool = Field(default=False, description="Enable secondary Wav2Vec2 ensemble")
    wav2vec2_model_id: str = "Hemgg/Deepfake-audio-detection"
    enable_torchscript: bool = Field(default=False, description="Use TorchScript tracing (False by default for stability on CPU)")

    # Paths to VoiceGuard original artifacts
    voiceguard_root: Path = _VOICEGUARD_DIR
    weights_dir: Path = _VOICEGUARD_DIR / "models" / "weights"
    config_dir: Path = _VOICEGUARD_DIR / "config"

    # Decision thresholds
    threshold_low: float = Field(default=0.30, ge=0.0, le=1.0, description="Scores below this are LIKELY HUMAN")
    threshold_high: float = Field(default=0.70, ge=0.0, le=1.0, description="Scores at or above this are LIKELY AI-GENERATED")

    # Audio validation limits
    max_upload_size_mb: int = Field(default=50, description="Maximum upload file size in MB")
    min_audio_duration_sec: float = Field(default=0.5, description="Minimum audio duration in seconds")
    max_audio_duration_sec: float = Field(default=180.0, description="Maximum audio duration in seconds")
    allowed_extensions: List[str] = ["wav", "mp3", "flac", "ogg", "pcm"]

    # Windowed analysis settings
    enable_windowed_analysis: bool = Field(default=True, description="Analyze full files via overlapping windows")
    window_duration_sec: float = Field(default=4.0375, description="AASIST model window length (64600 samples @ 16kHz)")
    hop_duration_sec: float = Field(default=2.0, description="Window hop length in seconds")

    # Streaming buffer settings
    streaming_buffer_sec: float = Field(default=6.0, description="Accumulation buffer duration for live streaming")
    streaming_overlap_sec: float = Field(default=2.0, description="Buffer overlap retention after inference")

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = AppSettings()
