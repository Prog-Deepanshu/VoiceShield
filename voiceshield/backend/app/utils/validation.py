"""
VoiceShield — Audio Validation Utilities
=========================================
Validation rules for secure audio file uploads, preventing path traversal,
buffer exhaustion, unsupported codecs, and corrupted streams.
"""

from pathlib import Path
from typing import Tuple
from fastapi import HTTPException, UploadFile
from app.config import settings

ALLOWED_MIME_PREFIXES = ("audio/", "application/octet-stream")
ALLOWED_EXTENSIONS = {f".{ext.lower()}" for ext in settings.allowed_extensions}


def sanitize_filename(filename: str) -> str:
    """Strip directory traversal elements and extract safe basename."""
    safe_name = Path(filename).name
    # Remove null bytes and strange control characters
    safe_name = "".join(c for c in safe_name if c.isprintable() and c not in '\\/:*?"<>|')
    return safe_name or "audio_upload.wav"


def validate_file_meta(file: UploadFile) -> str:
    """
    Validate file extension and MIME type before reading content.
    Returns sanitized filename.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload.")

    safe_name = sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format '{ext}'. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}"
        )

    # Validate content-type if provided
    content_type = (file.content_type or "").lower()
    if content_type and not any(content_type.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported MIME type '{content_type}'. Must be an audio file."
        )

    return safe_name


def validate_file_content(content: bytes, max_mb: int = settings.max_upload_size_mb) -> None:
    """Validate file content size."""
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty (0 bytes).")

    max_bytes = max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum limit of {max_mb} MB ({len(content)} bytes uploaded)."
        )


def validate_audio_duration(duration_sec: float) -> None:
    """Validate that decoded audio length meets minimum and maximum constraints."""
    if duration_sec < settings.min_audio_duration_sec:
        raise HTTPException(
            status_code=400,
            detail=f"Audio duration ({duration_sec:.2f}s) is too short. Minimum required: {settings.min_audio_duration_sec}s."
        )

    if duration_sec > settings.max_audio_duration_sec:
        raise HTTPException(
            status_code=400,
            detail=f"Audio duration ({duration_sec:.2f}s) exceeds maximum allowed ({settings.max_audio_duration_sec}s)."
        )
