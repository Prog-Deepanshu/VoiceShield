"""
VoiceGuard — Inference & Risk Classification Tests
=====================================================
"""

import numpy as np
import pytest

from core.ensemble import EnsembleScorer
from core.risk_classifier import RiskClassifier, RiskLevel, RecommendedAction


class TestEnsembleScorer:
    """Test the EnsembleScorer class."""

    def test_single_model_scoring(self):
        scorer = EnsembleScorer()
        result = scorer.score(
            aasist_result={"deepfake_score": 0.8, "latency_ms": 25.0}
        )
        assert result["deepfake_score"] == 0.8
        assert result["models_used"] == ["aasist"]
        assert "aasist" in result["model_scores"]

    def test_ensemble_scoring(self):
        scorer = EnsembleScorer(aasist_weight=0.7, wav2vec2_weight=0.3)
        result = scorer.score(
            aasist_result={"deepfake_score": 0.9, "latency_ms": 25.0},
            wav2vec2_result={"deepfake_score": 0.7, "latency_ms": 80.0},
        )
        expected = 0.7 * 0.9 + 0.3 * 0.7  # 0.63 + 0.21 = 0.84
        assert abs(result["deepfake_score"] - expected) < 0.01
        assert result["models_used"] == ["aasist", "wav2vec2"]
        assert result["total_latency_ms"] == 105.0

    def test_confidence_high_agreement(self):
        scorer = EnsembleScorer()
        result = scorer.score(
            aasist_result={"deepfake_score": 0.85, "latency_ms": 25.0},
            wav2vec2_result={"deepfake_score": 0.80, "latency_ms": 80.0},
        )
        # Models agree → high confidence
        assert result["confidence"] > 0.9

    def test_confidence_low_agreement(self):
        scorer = EnsembleScorer()
        result = scorer.score(
            aasist_result={"deepfake_score": 0.9, "latency_ms": 25.0},
            wav2vec2_result={"deepfake_score": 0.1, "latency_ms": 80.0},
        )
        # Models disagree → low confidence
        assert result["confidence"] < 0.3

    def test_single_model_confidence_extremes(self):
        scorer = EnsembleScorer()
        # Score near 1.0 → high confidence
        result = scorer.score({"deepfake_score": 0.95, "latency_ms": 25.0})
        assert result["confidence"] > 0.8

        # Score near 0.5 → low confidence
        result = scorer.score({"deepfake_score": 0.5, "latency_ms": 25.0})
        assert result["confidence"] < 0.1

    def test_weight_normalization(self):
        # Weights that don't sum to 1.0 should be normalized
        scorer = EnsembleScorer(aasist_weight=1.0, wav2vec2_weight=1.0)
        assert abs(scorer.aasist_weight - 0.5) < 0.01
        assert abs(scorer.wav2vec2_weight - 0.5) < 0.01


class TestRiskClassifier:
    """Test the RiskClassifier class."""

    @pytest.fixture
    def classifier(self):
        return RiskClassifier(threshold_low=0.3, threshold_high=0.7)

    def test_low_risk(self, classifier):
        risk = classifier.classify(0.1)
        assert risk.risk_level == RiskLevel.LOW
        assert risk.recommended_action == RecommendedAction.ALLOW
        assert not risk.should_alert

    def test_medium_risk(self, classifier):
        risk = classifier.classify(0.5)
        assert risk.risk_level == RiskLevel.MEDIUM
        assert risk.recommended_action == RecommendedAction.FLAG_FOR_REVIEW
        assert not risk.should_alert

    def test_high_risk(self, classifier):
        risk = classifier.classify(0.85)
        assert risk.risk_level == RiskLevel.HIGH
        assert risk.recommended_action == RecommendedAction.BLOCK_AND_ALERT
        assert risk.should_alert

    def test_boundary_low(self, classifier):
        # Exactly at threshold_low boundary → MEDIUM
        risk = classifier.classify(0.3)
        assert risk.risk_level == RiskLevel.MEDIUM

    def test_boundary_high(self, classifier):
        # Exactly at threshold_high boundary → HIGH
        risk = classifier.classify(0.7)
        assert risk.risk_level == RiskLevel.HIGH

    def test_edge_cases(self, classifier):
        # Score = 0 → LOW
        risk = classifier.classify(0.0)
        assert risk.risk_level == RiskLevel.LOW

        # Score = 1 → HIGH
        risk = classifier.classify(1.0)
        assert risk.risk_level == RiskLevel.HIGH

    def test_clamping(self, classifier):
        # Out of range scores should be clamped
        risk = classifier.classify(-0.5)
        assert risk.risk_level == RiskLevel.LOW

        risk = classifier.classify(1.5)
        assert risk.risk_level == RiskLevel.HIGH

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError):
            RiskClassifier(threshold_low=0.7, threshold_high=0.3)

    def test_to_dict(self, classifier):
        risk = classifier.classify(0.85)
        d = risk.to_dict()
        assert d["risk_level"] == "HIGH"
        assert d["recommended_action"] == "BLOCK_AND_ALERT"
        assert d["should_alert"] is True
        assert "description" in d

    def test_description_contains_score(self, classifier):
        risk = classifier.classify(0.85)
        assert "0.850" in risk.description
