"""
VoiceShield — FastAPI Application Entrypoint
============================================
Initializes FastAPI, configures CORS, handles model startup lifespan,
and mounts API routes.
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.config import settings
from app.services.inference_service import inference_service

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voiceshield")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: load models on startup and release on shutdown."""
    logger.info("=" * 60)
    logger.info("  🛡️  VoiceShield — AI Voice Deepfake Detection Service")
    logger.info("=" * 60)
    logger.info(f"Target model variant : {settings.model_variant}")
    logger.info(f"Inference device     : {settings.device}")
    logger.info(f"Decision thresholds  : LOW<{settings.threshold_low}, HIGH>={settings.threshold_high}")
    logger.info("-" * 60)

    try:
        inference_service.initialize()
        logger.info("✓ Model initialized and ready for real inference.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize VoiceGuard model: {e}", exc_info=True)

    yield

    logger.info("Shutting down VoiceShield...")


app = FastAPI(
    title="VoiceShield API",
    description="Enterprise Audio Deepfake Detection Wrapper around VoiceGuard AASIST",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "VoiceShield",
        "description": "AI Voice Deepfake Detection Service powered by VoiceGuard AASIST",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
        "model_info": "/api/model-info"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please check server logs."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
