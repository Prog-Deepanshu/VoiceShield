"""
VoiceGuard — Ensemble Scorer
==============================
Multi-model ensemble scoring that combines AASIST and Wav2Vec2
predictions into a single calibrated deepfake probability score.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EnsembleScorer:
    """
    Combines predictions from multiple deepfake detection models
    into a single calibrated risk score.

    Supports:
    - Single-model mode (AASIST only)
    - Dual-model ensemble (AASIST + Wav2Vec2) with configurable weights
    - Confidence-weighted averaging
    """

    def __init__(
        self,
        aasist_weight: float = 0.7,
        wav2vec2_weight: float = 0.3,
    ):
        """
        Args:
            aasist_weight:  Weight for AASIST model predictions
            wav2vec2_weight: Weight for Wav2Vec2 model predictions
        """
        self.aasist_weight = aasist_weight
        self.wav2vec2_weight = wav2vec2_weight

        # Validate weights sum to 1.0
        total = aasist_weight + wav2vec2_weight
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                f"Ensemble weights sum to {total}, normalizing to 1.0"
            )
            self.aasist_weight = aasist_weight / total
            self.wav2vec2_weight = wav2vec2_weight / total

    def score(
        self,
        aasist_result: dict,
        wav2vec2_result: Optional[dict] = None,
    ) -> dict:
        """
        Compute ensemble deepfake score.

        Args:
            aasist_result:  Output from AASISTInferenceEngine.infer()
            wav2vec2_result: Output from Wav2Vec2InferenceEngine.infer() (optional)

        Returns:
            Dictionary with:
                - deepfake_score: float [0, 1] (ensemble score)
                - confidence: float [0, 1] (model agreement)
                - models_used: list[str]
                - model_scores: dict of individual model scores
                - total_latency_ms: float
        """
        aasist_score = aasist_result["deepfake_score"]
        models_used = ["aasist"]
        model_scores = {"aasist": round(aasist_score, 4)}
        total_latency = aasist_result.get("latency_ms", 0)

        if wav2vec2_result is not None:
            # Weighted ensemble
            wav2vec2_score = wav2vec2_result["deepfake_score"]
            models_used.append("wav2vec2")
            model_scores["wav2vec2"] = round(wav2vec2_score, 4)
            total_latency += wav2vec2_result.get("latency_ms", 0)

            ensemble_score = (
                self.aasist_weight * aasist_score
                + self.wav2vec2_weight * wav2vec2_score
            )

            # Confidence: how much the models agree (1.0 = perfect agreement)
            agreement = 1.0 - abs(aasist_score - wav2vec2_score)
            confidence = agreement
        else:
            # Single model mode
            ensemble_score = aasist_score
            confidence = self._single_model_confidence(aasist_score)

        return {
            "deepfake_score": round(ensemble_score, 4),
            "confidence": round(confidence, 4),
            "models_used": models_used,
            "model_scores": model_scores,
            "total_latency_ms": round(total_latency, 2),
        }

    @staticmethod
    def _single_model_confidence(score: float) -> float:
        """
        Estimate confidence for single-model predictions.

        Scores near 0.0 or 1.0 indicate high confidence.
        Scores near 0.5 indicate low confidence (uncertain).
        """
        return abs(2 * score - 1.0)
