# =============================================================================
# VoiceGuard — Production Dockerfile
# =============================================================================
# Multi-stage build for optimized production image.
#
# Build:
#   docker build -t voiceguard:latest .
#
# Run:
#   docker run -p 8000:8000 -v ./models/weights:/app/models/weights voiceguard:latest
#
# GPU support:
#   docker run --gpus all -p 8000:8000 -v ./models/weights:/app/models/weights voiceguard:latest
# =============================================================================

# ── Stage 1: Base with system dependencies ───────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r voiceguard && useradd -r -g voiceguard -d /app -s /sbin/nologin voiceguard

WORKDIR /app

# ── Stage 2: Install Python dependencies ────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Production image ───────────────────────────────────────────
FROM dependencies AS production

# Copy application code
COPY config/ ./config/
COPY models/ ./models/
COPY core/ ./core/
COPY api/ ./api/
COPY services/ ./services/
COPY scripts/ ./scripts/

# Create log directory
RUN mkdir -p /app/logs && chown -R voiceguard:voiceguard /app

# Switch to non-root user
USER voiceguard

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
