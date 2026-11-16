"""
VoiceGuard — Settings Management
=================================
Pydantic-based configuration with environment variable overrides.
All settings can be configured via .env file or environment variables.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ModelVariant(str, Enum):
    AASIST = "aasist"
    AASIST_L = "aasist_l"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendedAction(str, Enum):
    ALLOW = "ALLOW"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    BLOCK_AND_ALERT = "BLOCK_AND_ALERT"


# Resolve project root (two levels up from config/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings with env-var overrides."""

    # ── Model Configuration ──────────────────────────────────────────
    model_variant: ModelVariant = Field(
        default=ModelVariant.AASIST_L,
        description="AASIST variant to use: 'aasist' (297K params) or 'aasist_l' (85K params)",
    )
    enable_ensemble: bool = Field(
        default=False,
        description="Enable Wav2Vec2 ensemble verification for higher accuracy",
    )
    wav2vec2_model_id: str = Field(
        default="Hemgg/Deepfake-audio-detection",
        description="HuggingFace model ID for the Wav2Vec2 deepfake detector",
    )

    # ── Audio Configuration ──────────────────────────────────────────
    sample_rate: int = Field(
        default=16000,
        description="Target sample rate in Hz (AASIST requires 16kHz)",
    )
    buffer_duration_sec: float = Field(
        default=6.0,
        ge=2.0,
        le=30.0,
        description="Seconds of audio to buffer before triggering analysis",
    )

    # ── Inference Configuration ──────────────────────────────────────
    device: str = Field(
        default="auto",
        description="Inference device: 'auto', 'cuda', 'cpu'",
    )
    max_latency_ms: int = Field(
        default=200,
        description="Target maximum inference latency in milliseconds",
    )
    enable_half_precision: bool = Field(
        default=False,
        description="Use FP16 inference on GPU for faster computation",
    )
    enable_torchscript: bool = Field(
        default=True,
        description="JIT-trace the model for optimized inference",
    )

    # ── Risk Thresholds ──────────────────────────────────────────────
    risk_threshold_low: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Scores below this are LOW risk",
    )
    risk_threshold_high: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Scores at or above this are HIGH risk",
    )

    # ── API Configuration ────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (None = disabled)",
    )
    cors_origins: str = Field(
        default="*",
        description="Comma-separated CORS origins",
    )

    # ── Alert Configuration ──────────────────────────────────────────
    webhook_url: Optional[str] = Field(
        default=None,
        description="Fraud alert webhook URL",
    )
    webhook_timeout_sec: int = Field(default=10)

    # ── Logging ──────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    audit_log_path: str = Field(default="./logs/audit.jsonl")

    # ── Derived Paths ────────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def model_config_path(self) -> Path:
        filename = f"{self.model_variant.value}.json"
        return _PROJECT_ROOT / "config" / filename

    @property
    def model_weights_path(self) -> Path:
        variant = self.model_variant.value.upper().replace("_", "-")
        return _PROJECT_ROOT / "models" / "weights" / f"{variant}.pth"

    @property
    def nb_samp(self) -> int:
        """Number of audio samples expected by the AASIST model."""
        return 64600

    @property
    def buffer_samples(self) -> int:
        """Number of samples in the audio buffer."""
        return int(self.sample_rate * self.buffer_duration_sec)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    def load_model_config(self) -> dict:
        """Load the AASIST model architecture configuration."""
        with open(self.model_config_path, "r") as f:
            return json.load(f)

    @field_validator("risk_threshold_high")
    @classmethod
    def validate_thresholds(cls, v, info):
        low = info.data.get("risk_threshold_low", 0.3)
        if v <= low:
            raise ValueError(
                f"risk_threshold_high ({v}) must be greater than risk_threshold_low ({low})"
            )
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton instance
settings = Settings()
