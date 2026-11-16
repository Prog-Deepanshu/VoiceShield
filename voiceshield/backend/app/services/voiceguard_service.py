"""
VoiceShield — VoiceGuard Integration Service
============================================
Direct wrapper around the underlying VoiceGuard AASIST detection engine.
Preserves the original model architecture and weights without modification.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from app.config import settings
from app.models.schemas import ClassificationState, ModelInfoResponse, SegmentAnalysis
from app.utils.preprocessing import (
    AASIST_NB_SAMP,
    TARGET_SAMPLE_RATE,
    pad_or_crop,
    resample_audio,
    slice_windows,
)

logger = logging.getLogger(__name__)


class VoiceGuardIntegrationService:
    """
    Adapter and lifecycle manager for the original VoiceGuard AASIST model.
    Encapsulates raw PyTorch operations and exposes standardized inference.
    """

    def __init__(self):
        self.device = self._resolve_device(settings.device)
        self.model: Optional[torch.nn.Module] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.weights_path: Optional[Path] = None
        self._is_loaded: bool = False
        self._load_duration_ms: float = 0.0
        self._parameter_count: int = 0
        self._ensure_voiceguard_in_path()

    def _ensure_voiceguard_in_path(self) -> None:
        """Add VoiceGuard root directory to sys.path so its native modules can be loaded."""
        vg_path = str(settings.voiceguard_root.resolve())
        if vg_path not in sys.path:
            sys.path.insert(0, vg_path)
            logger.info(f"Registered VoiceGuard in sys.path: {vg_path}")

    @staticmethod
    def _resolve_device(device_setting: str) -> torch.device:
        """Determine whether to use GPU or CPU based on hardware availability."""
        if device_setting == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def load_model(self) -> None:
        """Load the AASIST model architecture and pretrained weights from disk."""
        start_time = time.perf_counter()
        logger.info(f"Loading VoiceGuard model variant '{settings.model_variant}' on device '{self.device}'...")

        self._ensure_voiceguard_in_path()

        # Import original AASIST model class
        try:
            from models.aasist import Model as AASISTModel
        except ImportError as e:
            raise RuntimeError(f"Could not import VoiceGuard AASIST model from {settings.voiceguard_root}: {e}")

        # Load architecture configuration
        config_file = settings.config_dir / f"{settings.model_variant}.json"
        if not config_file.exists():
            raise FileNotFoundError(f"AASIST config not found at: {config_file}")

        with open(config_file, "r") as f:
            self.model_config = json.load(f)

        # Instantiate model architecture
        self.model = AASISTModel(self.model_config)

        # Resolve weights path
        variant_name = settings.model_variant.upper().replace("_", "-")
        self.weights_path = settings.weights_dir / f"{variant_name}.pth"

        if not self.weights_path.exists():
            # Check lowercase variant
            fallback_weights = settings.weights_dir / f"{settings.model_variant}.pth"
            if fallback_weights.exists():
                self.weights_path = fallback_weights

        if self.weights_path.exists():
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"✓ Loaded pretrained weights from {self.weights_path} ({self.weights_path.stat().st_size / 1024:.1f} KB)")
        else:
            logger.error(f"Pretrained weights missing at {self.weights_path}")
            raise FileNotFoundError(f"Missing pretrained weights at {self.weights_path}. Run scripts/download_weights.py first.")

        # Set evaluation mode and move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Calculate parameter count
        self._parameter_count = sum(p.numel() for p in self.model.parameters())
        self._load_duration_ms = (time.perf_counter() - start_time) * 1000

        # Run warmup
        self._warmup()

        self._is_loaded = True
        logger.info(
            f"✓ VoiceGuard inference service ready ({self._parameter_count:,} params, loaded in {self._load_duration_ms:.1f}ms)"
        )

    def _warmup(self, steps: int = 1) -> None:
        """Run warm-up passes to initialize PyTorch graphs and memory pools."""
        dummy = torch.randn(1, AASIST_NB_SAMP, device=self.device)
        with torch.no_grad():
            for _ in range(steps):
                self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def classify_score(self, score: float) -> Tuple[ClassificationState, float]:
        """
        Map a continuous deepfake probability score into one of three states:
        - LIKELY HUMAN (score < threshold_low)
        - UNCERTAIN (threshold_low <= score < threshold_high)
        - LIKELY AI-GENERATED (score >= threshold_high)

        Also computes a decision certainty metric (confidence).
        """
        # Confidence reflects distance from the maximum-entropy midpoint (0.5)
        # Score 0.0 or 1.0 -> confidence 1.0; Score 0.5 -> confidence 0.0
        certainty = round(abs(score - 0.5) * 2.0, 4)

        if score < settings.threshold_low:
            return ClassificationState.LIKELY_HUMAN, certainty
        elif score >= settings.threshold_high:
            return ClassificationState.LIKELY_AI_GENERATED, certainty
        else:
            return ClassificationState.UNCERTAIN, certainty

    def infer_window(self, audio_window: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single 64,600-sample (4.04s) audio window.

        Returns:
            Dict containing:
                - spoof_score: float [0, 1]
                - bonafide_score: float [0, 1]
                - deepfake_score: float [0, 1]
                - raw_logits: [float, float]
                - latency_ms: float
        """
        if not self._is_loaded:
            raise RuntimeError("VoiceGuard model is not loaded. Call load_model() first.")

        # Standardize amplitude to AASIST nominal operating range (peak 0.25).
        # This prevents SincConv front-end filterbank saturation and distortion
        # that otherwise misclassifies human conversational speech as synthetic.
        peak = float(np.max(np.abs(audio_window)))
        if peak > 1e-4:
            audio_window = (audio_window / peak) * 0.25

        # Ensure exact length (64,600 samples)
        audio_window = pad_or_crop(audio_window, AASIST_NB_SAMP)

        # Convert to tensor of shape (1, 64600)
        tensor = torch.from_numpy(audio_window).float().unsqueeze(0).to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            last_hidden, logits = self.model(tensor)

        # Convert logits via softmax: index 0 = spoof, index 1 = bonafide
        probs = F.softmax(logits, dim=1)
        spoof_prob = float(probs[0, 0].item())
        bonafide_prob = float(probs[0, 1].item())
        raw_logits = [float(logits[0, 0].item()), float(logits[0, 1].item())]

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "spoof_score": spoof_prob,
            "bonafide_score": bonafide_prob,
            "deepfake_score": spoof_prob,
            "raw_logits": raw_logits,
            "latency_ms": round(latency_ms, 2)
        }

    def analyze_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE,
        enable_windowing: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze an entire audio signal using the actual VoiceGuard model.
        Performs multi-window evaluation if audio exceeds single-window length.
        """
        start_time = time.perf_counter()

        # Resample to 16kHz if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = resample_audio(audio, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE

        total_samples = len(audio)
        duration_sec = total_samples / sample_rate

        # If audio is within single window duration or windowing disabled:
        if not enable_windowing or total_samples <= AASIST_NB_SAMP:
            single_result = self.infer_window(audio)
            score = single_result["deepfake_score"]
            state, confidence = self.classify_score(score)

            segment = SegmentAnalysis(
                window_index=0,
                start_sec=0.0,
                end_sec=round(duration_sec, 2),
                score=round(score, 4),
                classification=state,
                spoof_score=round(single_result["spoof_score"], 4),
                bonafide_score=round(single_result["bonafide_score"], 4)
            )

            total_latency = (time.perf_counter() - start_time) * 1000

            return {
                "model_score": round(score, 4),
                "classification": state,
                "confidence": confidence,
                "raw_logits": single_result["raw_logits"],
                "segments": [segment],
                "processing_ms": round(total_latency, 2),
                "is_uncertain": state == ClassificationState.UNCERTAIN,
                "uncertainty_reason": "Score fell within the intermediate uncertainty band (0.30 - 0.70)" if state == ClassificationState.UNCERTAIN else None
            }

        # Multi-window evaluation
        windows = slice_windows(audio, sample_rate=sample_rate)
        segments: List[SegmentAnalysis] = []
        scores: List[float] = []
        raw_logits_last = None

        for idx, start_sec, end_sec, chunk in windows:
            res = self.infer_window(chunk)
            w_score = res["deepfake_score"]
            w_state, _ = self.classify_score(w_score)
            raw_logits_last = res["raw_logits"]
            scores.append(w_score)

            segments.append(
                SegmentAnalysis(
                    window_index=idx,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    score=round(w_score, 4),
                    classification=w_state,
                    spoof_score=round(res["spoof_score"], 4),
                    bonafide_score=round(res["bonafide_score"], 4)
                )
            )

        # Aggregation:
        # A robust combination of median and mean score provides resilient whole-recording representation,
        # preventing a single transient breath or syllable from falsely triggering deepfake classification.
        median_score = float(np.median(scores))
        mean_score = float(np.mean(scores))
        aggregated_score = round(0.7 * median_score + 0.3 * mean_score, 4)

        final_state, confidence = self.classify_score(aggregated_score)
        total_latency = (time.perf_counter() - start_time) * 1000

        is_uncertain = final_state == ClassificationState.UNCERTAIN
        uncertainty_reason = None
        if is_uncertain:
            uncertainty_reason = "Model score lies in the decision boundary region (0.30 - 0.70) without strong evidence in either direction."

        return {
            "model_score": aggregated_score,
            "classification": final_state,
            "confidence": confidence,
            "raw_logits": raw_logits_last,
            "segments": segments,
            "processing_ms": round(total_latency, 2),
            "is_uncertain": is_uncertain,
            "uncertainty_reason": uncertainty_reason
        }

    def get_model_info(self) -> ModelInfoResponse:
        """Return comprehensive metadata about the model architecture and training."""
        return ModelInfoResponse(
            architecture="AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)",
            model_variant=settings.model_variant,
            parameter_count=self._parameter_count,
            input_requirements={
                "sample_rate_hz": 16000,
                "channels": 1,
                "window_samples": AASIST_NB_SAMP,
                "window_duration_sec": round(AASIST_NB_SAMP / TARGET_SAMPLE_RATE, 3),
                "dtype": "float32 [-1.0, 1.0]"
            },
            weights_path=str(self.weights_path) if self.weights_path else "",
            weights_found=bool(self.weights_path and self.weights_path.exists()),
            available_methods=["AASIST-L (85K params)", "AASIST full (297K params)", "Wav2Vec2 Ensemble (optional)"],
            supported_inference_modes=[
                "Single-file windowed inference",
                "Segmented multi-window timeline analysis",
                "Sliding buffer streaming inference"
            ],
            decision_thresholds={
                "likely_human_upper_bound": settings.threshold_low,
                "uncertain_lower_bound": settings.threshold_low,
                "uncertain_upper_bound": settings.threshold_high,
                "likely_ai_lower_bound": settings.threshold_high
            },
            calibration_method="Softmax over raw AASIST logits [spoof, bonafide] followed by thresholded 3-tier risk classification.",
            important_limitations=[
                "Model was trained on the ASVspoof 2019 Logical Access (LA) clean benchmark dataset.",
                "High-frequency studio audio or modern aggressive lossy compression (e.g. cellular AMR-WB, low-bitrate MP3) can produce out-of-distribution acoustic shifts.",
                "Background acoustic noise or music may trigger elevated spoof scores; scores should be corroborated with spectral metrics.",
                "A classification of LIKELY AI-GENERATED indicates synthetic speech acoustic markers, but is NOT standalone legal proof of malicious fraud."
            ]
        )
