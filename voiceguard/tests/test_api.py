"""
VoiceGuard — API Integration Tests
=====================================
"""

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


def _create_test_wav(duration_sec: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Create a test WAV file in memory."""
    samples = int(sample_rate * duration_sec)
    audio = np.random.randn(samples).astype(np.float32) * 0.5
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture(scope="module")
def test_client():
    """Create a test client. Skips if models can't be loaded."""
    try:
        from api.app import app
        with TestClient(app) as client:
            yield client
    except Exception as e:
        pytest.skip(f"Cannot create test client (models may not be available): {e}")


class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "VoiceGuard"
        assert "version" in data


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "gpu_available" in data
        assert "version" in data

    def test_readiness(self, test_client):
        response = test_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data

    def test_metrics(self, test_client):
        response = test_client.get("/health/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_analyses" in data
        assert "avg_latency_ms" in data


class TestAnalyzeEndpoint:
    """Test the audio analysis endpoint."""

    def test_analyze_wav(self, test_client):
        """Test analysis of a WAV file."""
        wav_data = _create_test_wav(duration_sec=3.0)
        response = test_client.post(
            "/api/v1/analyze",
            files={"file": ("test.wav", wav_data, "audio/wav")},
            data={"call_id": "TEST-001"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "deepfake_score" in data
        assert 0.0 <= data["deepfake_score"] <= 1.0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert data["recommended_action"] in ["ALLOW", "FLAG_FOR_REVIEW", "BLOCK_AND_ALERT"]
        assert data["call_id"] == "TEST-001"
        assert "inference_latency_ms" in data
        assert len(data["models_used"]) > 0

    def test_analyze_auto_call_id(self, test_client):
        """Test that call_id is auto-generated when not provided."""
        wav_data = _create_test_wav(duration_sec=3.0)
        response = test_client.post(
            "/api/v1/analyze",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["call_id"].startswith("UPLOAD-")

    def test_analyze_empty_file(self, test_client):
        """Test rejection of empty files."""
        response = test_client.post(
            "/api/v1/analyze",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400

    def test_analyze_spectral_data(self, test_client):
        """Test that spectral analysis is included in response."""
        wav_data = _create_test_wav(duration_sec=4.0)
        response = test_client.post(
            "/api/v1/analyze",
            files={"file": ("test.wav", wav_data, "audio/wav")},
            data={"call_id": "TEST-SPECTRAL"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "spectral_analysis" in data
        if data["spectral_analysis"]:
            assert "spectral_centroid_hz" in data["spectral_analysis"]
            assert "pitch_mean_hz" in data["spectral_analysis"]
