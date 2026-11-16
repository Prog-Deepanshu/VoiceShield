"""
VoiceShield — High-Level Inference Service
==========================================
Coordinates audio ingestion, model lifecycle, concurrency controls,
in-memory history storage for dashboard analytics, and batch processing.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.models.schemas import (
    AnalysisResult,
    BatchAnalysisResult,
    HealthStatus,
    ModelInfoResponse,
)
from app.services.audio_service import AudioService
from app.services.voiceguard_service import VoiceGuardIntegrationService

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Central coordinator for VoiceShield.
    Maintains model lifecycle, local analysis history, and safe concurrency limits.
    """

    def __init__(self):
        self.voiceguard_service = VoiceGuardIntegrationService()
        self.audio_service = AudioService()
        # Maximum simultaneous inference operations to protect system stability
        self._inference_semaphore = asyncio.Semaphore(4)
        self._start_time = datetime.utcnow()
        # In-memory history of analyses (capped to 500 recent items)
        self._history: List[AnalysisResult] = []
        self._history_lock = asyncio.Lock()

    def initialize(self) -> None:
        """Load the model during application startup."""
        self.voiceguard_service.load_model()

    @property
    def is_ready(self) -> bool:
        return self.voiceguard_service.is_loaded

    def get_health_status(self) -> HealthStatus:
        """Construct system health snapshot."""
        import torch

        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None

        return HealthStatus(
            status="healthy" if self.is_ready else "degraded",
            model_loaded=self.is_ready,
            model_variant=settings.model_variant,
            device=str(self.voiceguard_service.device),
            cuda_available=cuda_avail,
            gpu_name=gpu_name,
            torch_version=torch.__version__,
            uptime_sec=round(uptime, 1),
            version=settings.app_version,
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Return full model architecture and limitations info."""
        return self.voiceguard_service.get_model_info()

    async def analyze_file(
        self,
        file_bytes: bytes,
        filename: str,
        enable_windowing: bool = True
    ) -> AnalysisResult:
        """
        Process and classify an uploaded audio file.
        Runs under concurrency semaphore to prevent event loop starvation.
        """
        # Decode audio and extract metadata
        audio_array, sr, metadata = self.audio_service.decode_audio_bytes(file_bytes, filename)

        # Extract spectral diagnostics in background or worker thread
        loop = asyncio.get_running_loop()
        diagnostics = await loop.run_in_executor(
            None, self.audio_service.extract_diagnostics, audio_array, sr
        )

        # Run inference protected by concurrency semaphore
        async with self._inference_semaphore:
            # Offload synchronous PyTorch model inference to default threadpool executor
            detection_res = await loop.run_in_executor(
                None,
                self.voiceguard_service.analyze_audio,
                audio_array,
                sr,
                enable_windowing
            )

        analysis_id = f"ANL-{uuid.uuid4().hex[:10].upper()}"

        result = AnalysisResult(
            id=analysis_id,
            filename=filename,
            classification=detection_res["classification"],
            model_score=detection_res["model_score"],
            confidence=detection_res["confidence"],
            raw_logits=detection_res.get("raw_logits"),
            model_name="VoiceGuard-AASIST",
            model_version=f"{settings.model_variant}-v1.0",
            audio_metadata=metadata,
            processing_duration_ms=detection_res["processing_ms"],
            created_at=datetime.utcnow(),
            segments=detection_res["segments"],
            spectral_diagnostics=diagnostics,
            is_uncertain=detection_res["is_uncertain"],
            uncertainty_reason=detection_res.get("uncertainty_reason")
        )

        # Append to local history
        async with self._history_lock:
            self._history.insert(0, result)
            if len(self._history) > 500:
                self._history.pop()

        return result

    async def analyze_batch(
        self,
        files_data: List[tuple[str, bytes]]
    ) -> BatchAnalysisResult:
        """Process multiple audio files concurrently."""
        tasks = [
            self.analyze_file(file_bytes=content, filename=name)
            for name, content in files_data
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        human_cnt = sum(1 for r in results if r.classification.value == "LIKELY HUMAN")
        ai_cnt = sum(1 for r in results if r.classification.value == "LIKELY AI-GENERATED")
        uncertain_cnt = sum(1 for r in results if r.classification.value == "UNCERTAIN")

        total_dur = sum(r.audio_metadata.duration_sec for r in results)
        total_proc = sum(r.processing_duration_ms for r in results)

        return BatchAnalysisResult(
            total_files=len(results),
            human_count=human_cnt,
            ai_generated_count=ai_cnt,
            uncertain_count=uncertain_cnt,
            total_duration_sec=round(total_dur, 2),
            total_processing_ms=round(total_proc, 2),
            results=results
        )

    async def get_history(self, limit: int = 100) -> List[AnalysisResult]:
        """Return recent analysis history."""
        async with self._history_lock:
            return list(self._history[:limit])

    async def clear_history(self) -> None:
        """Clear local in-memory history."""
        async with self._history_lock:
            self._history.clear()


# Global singleton instance
inference_service = InferenceService()
