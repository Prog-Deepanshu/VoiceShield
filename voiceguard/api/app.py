"""
VoiceGuard — FastAPI Application Factory
==========================================
Creates and configures the FastAPI application with all routes,
middleware, model lifecycle management, and error handling.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware import RequestLoggingMiddleware
from api.routes import analyze, health, stream
from config.settings import settings
from core.ensemble import EnsembleScorer
from core.inference_engine import AASISTInferenceEngine, Wav2Vec2InferenceEngine
from core.risk_classifier import RiskClassifier
from services.alert_service import AlertService
from services.audit_logger import AuditLogger

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voiceguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads models on startup and releases resources on shutdown.
    """
    logger.info("=" * 60)
    logger.info("  VoiceGuard — AI Voice Deepfake Detection Service")
    logger.info("=" * 60)
    logger.info(f"Model variant : {settings.model_variant.value}")
    logger.info(f"Ensemble mode : {'ENABLED' if settings.enable_ensemble else 'DISABLED'}")
    logger.info(f"Device target : {settings.device}")
    logger.info(f"Buffer duration: {settings.buffer_duration_sec}s")
    logger.info(f"Risk thresholds: LOW<{settings.risk_threshold_low}, HIGH≥{settings.risk_threshold_high}")
    logger.info("-" * 60)

    # ── Store settings in app state ──────────────────────────────────
    app.state.settings = settings
    app.state.start_time = time.time()
    app.state.total_analyses = 0
    app.state.total_high_risk = 0
    app.state.total_medium_risk = 0
    app.state.avg_latency_ms = 0.0
    app.state.min_latency_ms = float("inf")
    app.state.max_latency_ms = 0.0

    # ── Load AASIST model ────────────────────────────────────────────
    logger.info("Loading AASIST inference engine...")
    aasist_engine = AASISTInferenceEngine(settings)
    aasist_engine.load()
    app.state.aasist_engine = aasist_engine
    app.state.device = aasist_engine.device

    # ── Load Wav2Vec2 model (optional) ───────────────────────────────
    if settings.enable_ensemble:
        logger.info("Loading Wav2Vec2 ensemble engine...")
        wav2vec2_engine = Wav2Vec2InferenceEngine(settings.wav2vec2_model_id)
        wav2vec2_engine.load()
        app.state.wav2vec2_engine = wav2vec2_engine
    else:
        app.state.wav2vec2_engine = None

    # ── Initialize scoring & classification ──────────────────────────
    app.state.ensemble_scorer = EnsembleScorer()
    app.state.risk_classifier = RiskClassifier(
        threshold_low=settings.risk_threshold_low,
        threshold_high=settings.risk_threshold_high,
    )

    # ── Initialize services ──────────────────────────────────────────
    app.state.alert_service = AlertService(
        webhook_url=settings.webhook_url,
        timeout_sec=settings.webhook_timeout_sec,
    )
    app.state.audit_logger = AuditLogger(
        log_path=settings.audit_log_path,
    )

    logger.info("-" * 60)
    logger.info("✓ VoiceGuard is READY for inference")
    logger.info(f"  API: http://{settings.api_host}:{settings.api_port}")
    logger.info(f"  Docs: http://{settings.api_host}:{settings.api_port}/docs")
    logger.info("=" * 60)

    yield  # ← Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down VoiceGuard...")
    app.state.audit_logger.close()
    logger.info("Goodbye.")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="VoiceGuard API",
        description=(
            "Real-time AI voice deepfake detection service for banking call centers. "
            "Analyzes live call audio using AASIST and optionally Wav2Vec2 to detect "
            "synthetic speech within seconds."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS Middleware ──────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Logging Middleware ────────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)

    # ── Routes ───────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(stream.router)

    # ── Root endpoint ────────────────────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "service": "VoiceGuard",
            "description": "AI Voice Deepfake Detection for Banking Call Centers",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    # ── Global Exception Handler ─────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.log_level == "DEBUG" else "An unexpected error occurred",
            },
        )

    return app


# Application instance (used by uvicorn)
app = create_app()
