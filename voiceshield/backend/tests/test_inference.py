"""
Integration tests for VoiceGuard model loading and real inference.
"""

from pathlib import Path
import numpy as np
import pytest

from app.config import settings
from app.models.schemas import ClassificationState
from app.services.voiceguard_service import VoiceGuardIntegrationService


@pytest.fixture(scope="module")
def vg_service():
    service = VoiceGuardIntegrationService()
    service.load_model()
    return service


def test_model_loaded(vg_service):
    assert vg_service.is_loaded
    assert vg_service.parameter_count > 0
    assert vg_service.model is not None


def test_infer_window_shape(vg_service):
    # AASIST input window of 64,600 samples
    dummy = np.random.randn(64600).astype(np.float32)
    res = vg_service.infer_window(dummy)

    assert "deepfake_score" in res
    assert "bonafide_score" in res
    assert "raw_logits" in res
    assert 0.0 <= res["deepfake_score"] <= 1.0
    assert 0.0 <= res["bonafide_score"] <= 1.0
    assert len(res["raw_logits"]) == 2
    assert res["latency_ms"] > 0


def test_classify_score(vg_service):
    # Low score -> LIKELY HUMAN
    state, conf = vg_service.classify_score(0.15)
    assert state == ClassificationState.LIKELY_HUMAN
    assert conf > 0.5

    # Intermediate score -> UNCERTAIN
    state, conf = vg_service.classify_score(0.50)
    assert state == ClassificationState.UNCERTAIN
    assert conf == pytest.approx(0.0, 0.01)

    # High score -> LIKELY AI-GENERATED
    state, conf = vg_service.classify_score(0.85)
    assert state == ClassificationState.LIKELY_AI_GENERATED
    assert conf > 0.5


def test_real_sample_inference(vg_service):
    sample_file = settings.voiceguard_root / "samples" / "synthetic_1.mp3"
    if not sample_file.exists():
        pytest.skip("synthetic_1.mp3 sample not found in voiceguard/samples")

    import soundfile as sf
    audio, sr = sf.read(str(sample_file))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Run multi-window analysis on synthetic audio
    result = vg_service.analyze_audio(audio, sample_rate=sr, enable_windowing=True)

    assert "model_score" in result
    assert "classification" in result
    assert "segments" in result
    assert len(result["segments"]) > 0
    assert 0.0 <= result["model_score"] <= 1.0
    assert result["processing_ms"] > 0
