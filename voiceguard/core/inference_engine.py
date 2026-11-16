"""
VoiceGuard — Inference Engine
==============================
Model inference orchestration for AASIST and Wav2Vec2 deepfake detectors.

Handles model loading, device management, optional TorchScript optimization,
CUDA warmup, and standardized inference output.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config.settings import Settings
from core.preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)


class AASISTInferenceEngine:
    """
    Inference engine for the AASIST deepfake detection model.

    Loads pretrained weights, configures the model for inference-only
    operation, and provides a standardized inference interface.

    The model outputs two logits:
        - output[:, 0] → spoof/synthetic score
        - output[:, 1] → bonafide/human score

    We convert to a deepfake probability via softmax.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the AASIST inference engine.

        Args:
            settings: Application settings containing model config and paths
        """
        self.settings = settings
        self.device = self._resolve_device(settings.device)
        self.model = None
        self.preprocessor = AudioPreprocessor(
            target_sr=settings.sample_rate,
            nb_samp=settings.nb_samp,
        )
        self._is_loaded = False

    def load(self) -> None:
        """Load the AASIST model and pretrained weights."""
        from models.aasist import Model as AASISTModel

        model_config = self.settings.load_model_config()
        logger.info(
            f"Loading AASIST model variant={self.settings.model_variant.value} "
            f"on device={self.device}"
        )

        # Initialize model architecture
        self.model = AASISTModel(model_config)

        # Load pretrained weights
        weights_path = self.settings.model_weights_path
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights from {weights_path}")
        else:
            logger.warning(
                f"Pretrained weights not found at {weights_path}. "
                "Model will use random initialization. "
                "Run `python scripts/download_weights.py` to download weights."
            )

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Count parameters
        nb_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded with {nb_params:,} parameters")

        # Optional: Half precision for GPU
        if self.settings.enable_half_precision and self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Half precision (FP16) enabled")

        # Optional: TorchScript optimization
        if self.settings.enable_torchscript:
            self._apply_torchscript()

        # CUDA warmup — run dummy inferences to prime kernels
        self._warmup()

        self._is_loaded = True
        logger.info("AASIST inference engine ready")

    def infer(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Run deepfake detection inference on audio.

        Args:
            audio:       Float32 numpy array of audio samples
            sample_rate: Sample rate of the input audio

        Returns:
            Dictionary with:
                - deepfake_score: float [0, 1] (0=human, 1=synthetic)
                - bonafide_score: float [0, 1]
                - label: str ("BONAFIDE" or "SPOOF")
                - latency_ms: float
                - embedding: list[float] (last hidden representation)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Preprocess audio
        audio_tensor = self.preprocessor.preprocess(audio, sample_rate)

        # Ensure batch dimension: (1, nb_samp)
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device)

        # Half precision if enabled
        if self.settings.enable_half_precision and self.device.type == "cuda":
            audio_tensor = audio_tensor.half()

        # Inference
        with torch.no_grad():
            last_hidden, output = self.model(audio_tensor)

        # Convert logits to probabilities
        probs = F.softmax(output, dim=1)
        deepfake_score = probs[0, 0].item()  # spoof probability
        bonafide_score = probs[0, 1].item()   # bonafide probability

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "deepfake_score": deepfake_score,
            "bonafide_score": bonafide_score,
            "label": "SPOOF" if deepfake_score > bonafide_score else "BONAFIDE",
            "latency_ms": round(latency_ms, 2),
            "embedding": last_hidden[0].cpu().float().numpy().tolist(),
        }

    def infer_tensor(self, audio_tensor: torch.Tensor) -> dict:
        """
        Run inference on a pre-processed tensor (skips preprocessing).

        Args:
            audio_tensor: Preprocessed tensor of shape (nb_samp,) or (1, nb_samp)

        Returns:
            Same as infer()
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        audio_tensor = audio_tensor.to(self.device)

        if self.settings.enable_half_precision and self.device.type == "cuda":
            audio_tensor = audio_tensor.half()

        with torch.no_grad():
            last_hidden, output = self.model(audio_tensor)

        probs = F.softmax(output, dim=1)
        deepfake_score = probs[0, 0].item()
        bonafide_score = probs[0, 1].item()

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "deepfake_score": deepfake_score,
            "bonafide_score": bonafide_score,
            "label": "SPOOF" if deepfake_score > bonafide_score else "BONAFIDE",
            "latency_ms": round(latency_ms, 2),
            "embedding": last_hidden[0].cpu().float().numpy().tolist(),
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _apply_torchscript(self) -> None:
        """Apply TorchScript tracing for optimized inference."""
        try:
            dummy_input = torch.randn(1, self.settings.nb_samp).to(self.device)
            if self.settings.enable_half_precision and self.device.type == "cuda":
                dummy_input = dummy_input.half()

            self.model = torch.jit.trace(self.model, dummy_input)
            logger.info("TorchScript tracing applied successfully")
        except Exception as e:
            logger.warning(
                f"TorchScript tracing failed, falling back to eager mode: {e}"
            )

    def _warmup(self, n_warmup: int = 3) -> None:
        """Run dummy inferences to prime CUDA kernels and JIT caches."""
        logger.info(f"Running {n_warmup} warmup inferences...")
        dummy = torch.randn(1, self.settings.nb_samp).to(self.device)
        if self.settings.enable_half_precision and self.device.type == "cuda":
            dummy = dummy.half()

        for _ in range(n_warmup):
            with torch.no_grad():
                self.model(dummy)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        logger.info("Warmup complete")

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)


class Wav2Vec2InferenceEngine:
    """
    Inference engine for the Wav2Vec2-based deepfake detection model
    from HuggingFace (Hemgg/Deepfake-audio-detection).

    Used as an optional ensemble verifier alongside AASIST.
    """

    def __init__(self, model_id: str = "Hemgg/Deepfake-audio-detection"):
        """
        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id
        self.model = None
        self.feature_extractor = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._is_loaded = False
        self._label_map: dict[int, str] = {}

    def load(self) -> None:
        """Load the Wav2Vec2 model from HuggingFace."""
        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )
        except ImportError:
            raise ImportError(
                "transformers library required for Wav2Vec2 ensemble. "
                "Install with: pip install transformers"
            )

        logger.info(f"Loading Wav2Vec2 model: {self.model_id}")

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_id
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Build label map
        if hasattr(self.model.config, "id2label"):
            self._label_map = self.model.config.id2label

        nb_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Wav2Vec2 model loaded with {nb_params:,} parameters, "
            f"labels={self._label_map}"
        )

        # Warmup
        self._warmup()
        self._is_loaded = True

    def infer(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Run deepfake detection with Wav2Vec2.

        Args:
            audio:       Float32 numpy array
            sample_rate: Sample rate

        Returns:
            Dictionary with:
                - deepfake_score: float [0, 1]
                - label: str
                - latency_ms: float
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Resample to 16kHz if necessary
        target_sr = 16000
        if sample_rate != target_sr:
            import torch
            import torchaudio
            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            )
            audio = resampler(tensor).squeeze(0).numpy()
            sample_rate = target_sr

        # Extract features using HuggingFace feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        # Determine which class index represents "fake"
        fake_idx = self._get_fake_idx()
        deepfake_score = probs[0, fake_idx].item() if fake_idx is not None else 0.5

        predicted_idx = torch.argmax(probs, dim=1).item()
        predicted_label = self._label_map.get(predicted_idx, f"class_{predicted_idx}")

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "deepfake_score": deepfake_score,
            "label": predicted_label,
            "latency_ms": round(latency_ms, 2),
            "all_scores": {
                self._label_map.get(i, f"class_{i}"): round(probs[0, i].item(), 4)
                for i in range(probs.shape[1])
            },
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def _get_fake_idx(self) -> Optional[int]:
        """Find the class index corresponding to 'fake' or 'spoof'."""
        for idx, label in self._label_map.items():
            if any(
                keyword in label.lower()
                for keyword in ["fake", "spoof", "synthetic", "deepfake"]
            ):
                return idx
        # If no explicit fake label, assume binary: class 0 = fake
        return 0

    def _warmup(self, n_warmup: int = 2) -> None:
        """Warmup inference."""
        logger.info("Running Wav2Vec2 warmup inferences...")
        dummy = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds

        inputs = self.feature_extractor(
            dummy, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        for _ in range(n_warmup):
            with torch.no_grad():
                self.model(**inputs)

        logger.info("Wav2Vec2 warmup complete")
