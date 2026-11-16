"""
VoiceGuard — Risk Classifier
==============================
Maps ensemble deepfake scores to risk levels and recommended actions
for the banking fraud prevention pipeline.
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendedAction(str, Enum):
    """Recommended actions for each risk level."""
    ALLOW = "ALLOW"
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"
    BLOCK_AND_ALERT = "BLOCK_AND_ALERT"


@dataclass
class RiskAssessment:
    """Complete risk assessment for a call analysis."""
    risk_level: RiskLevel
    recommended_action: RecommendedAction
    description: str
    should_alert: bool

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level.value,
            "recommended_action": self.recommended_action.value,
            "description": self.description,
            "should_alert": self.should_alert,
        }


class RiskClassifier:
    """
    Classifies deepfake detection scores into actionable risk levels.

    Default thresholds:
        - score < 0.3   → LOW    → ALLOW
        - 0.3 ≤ score < 0.7 → MEDIUM → FLAG_FOR_REVIEW
        - score ≥ 0.7   → HIGH   → BLOCK_AND_ALERT

    Thresholds are configurable to tune sensitivity:
        - Lower thresholds = more aggressive (more false positives)
        - Higher thresholds = more conservative (fewer false positives)
    """

    def __init__(
        self,
        threshold_low: float = 0.3,
        threshold_high: float = 0.7,
    ):
        """
        Args:
            threshold_low:  Scores below this are LOW risk
            threshold_high: Scores at or above this are HIGH risk
        """
        if threshold_high <= threshold_low:
            raise ValueError(
                f"threshold_high ({threshold_high}) must be greater than "
                f"threshold_low ({threshold_low})"
            )

        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def classify(self, deepfake_score: float, confidence: float = 1.0) -> RiskAssessment:
        """
        Classify a deepfake score into a risk assessment.

        Args:
            deepfake_score: Ensemble deepfake probability [0, 1]
            confidence:     Model confidence [0, 1]

        Returns:
            RiskAssessment with level, action, and description
        """
        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, deepfake_score))

        if score < self.threshold_low:
            return RiskAssessment(
                risk_level=RiskLevel.LOW,
                recommended_action=RecommendedAction.ALLOW,
                description=(
                    f"Voice appears authentic (score: {score:.3f}). "
                    f"No synthetic speech artifacts detected. "
                    f"Confidence: {confidence:.1%}"
                ),
                should_alert=False,
            )

        elif score < self.threshold_high:
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                recommended_action=RecommendedAction.FLAG_FOR_REVIEW,
                description=(
                    f"Uncertain detection result (score: {score:.3f}). "
                    f"Some spectral anomalies detected that may indicate "
                    f"voice manipulation or poor audio quality. "
                    f"Manual review recommended. Confidence: {confidence:.1%}"
                ),
                should_alert=False,
            )

        else:
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                recommended_action=RecommendedAction.BLOCK_AND_ALERT,
                description=(
                    f"High probability of synthetic speech detected "
                    f"(score: {score:.3f}). Voice exhibits artifacts "
                    f"consistent with AI-generated audio (vocoder patterns, "
                    f"pitch smoothing, harmonic inconsistencies). "
                    f"Immediate investigation recommended. "
                    f"Confidence: {confidence:.1%}"
                ),
                should_alert=True,
            )
