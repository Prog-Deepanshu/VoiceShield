"""
VoiceGuard — Health Check Routes
==================================
Service health, readiness, and metrics endpoints.
"""

import time

import torch
from fastapi import APIRouter, Request

from api.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns the current health status of the VoiceGuard service.",
)
async def health_check(request: Request) -> HealthResponse:
    """Basic health check — returns service status and loaded models."""
    app = request.app

    models_loaded = []
    if hasattr(app.state, "aasist_engine") and app.state.aasist_engine.is_loaded:
        models_loaded.append(f"aasist ({app.state.settings.model_variant.value})")
    if hasattr(app.state, "wav2vec2_engine") and app.state.wav2vec2_engine is not None:
        if app.state.wav2vec2_engine.is_loaded:
            models_loaded.append("wav2vec2")

    uptime = time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0

    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        gpu_available=torch.cuda.is_available(),
        device=str(app.state.device) if hasattr(app.state, "device") else "unknown",
        avg_latency_ms=app.state.avg_latency_ms if hasattr(app.state, "avg_latency_ms") else 0.0,
        total_analyses=app.state.total_analyses if hasattr(app.state, "total_analyses") else 0,
        uptime_seconds=round(uptime, 1),
        version="1.0.0",
    )


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Returns 200 if models are loaded and ready for inference.",
)
async def readiness_check(request: Request) -> dict:
    """Kubernetes-compatible readiness probe."""
    app = request.app

    ready = (
        hasattr(app.state, "aasist_engine")
        and app.state.aasist_engine.is_loaded
    )

    if not ready:
        return {"ready": False, "reason": "Models not yet loaded"}

    return {"ready": True}


@router.get(
    "/metrics",
    summary="Inference metrics",
    description="Returns inference statistics for monitoring.",
)
async def metrics(request: Request) -> dict:
    """Prometheus-compatible metrics endpoint."""
    app = request.app

    return {
        "total_analyses": getattr(app.state, "total_analyses", 0),
        "total_high_risk_alerts": getattr(app.state, "total_high_risk", 0),
        "total_medium_risk_flags": getattr(app.state, "total_medium_risk", 0),
        "avg_latency_ms": round(getattr(app.state, "avg_latency_ms", 0.0), 2),
        "min_latency_ms": round(getattr(app.state, "min_latency_ms", 0.0), 2),
        "max_latency_ms": round(getattr(app.state, "max_latency_ms", 0.0), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_used_mb": (
            round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            if torch.cuda.is_available()
            else 0
        ),
    }
