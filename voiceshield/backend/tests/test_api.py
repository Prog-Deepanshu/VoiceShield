"""
End-to-end API integration tests using FastAPI TestClient.
"""

import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.services.inference_service import inference_service


@pytest.fixture(scope="module")
def client():
    # Ensure model is initialized
    if not inference_service.is_ready:
        inference_service.initialize()
    with TestClient(app) as test_client:
        yield test_client


def _make_wav_bytes(duration_sec: float = 2.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sine = 0.4 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, sine, sr, format="WAV")
    return bio.getvalue()


def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "VoiceShield"


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["model_loaded"] is True
    assert "torch_version" in data


def test_model_info_endpoint(client):
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "AASIST" in data["architecture"]
    assert data["parameter_count"] > 0
    assert len(data["important_limitations"]) > 0


def test_analyze_audio_valid(client):
    wav = _make_wav_bytes(duration_sec=2.5)
    files = {"file": ("test_call.wav", wav, "audio/wav")}
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 200
    data = res.json()

    assert data["classification"] in ("LIKELY HUMAN", "LIKELY AI-GENERATED", "UNCERTAIN")
    assert 0.0 <= data["model_score"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["audio_metadata"]["filename"] == "test_call.wav"
    assert data["audio_metadata"]["duration_sec"] == pytest.approx(2.5, 0.2)
    assert data["spectral_diagnostics"] is not None
    assert len(data["segments"]) > 0


def test_analyze_empty_file_rejected(client):
    files = {"file": ("empty.wav", b"", "audio/wav")}
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 400


def test_analyze_unsupported_extension(client):
    files = {"file": ("script.sh", b"echo hello", "text/plain")}
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 415


def test_waveform_endpoint(client):
    wav = _make_wav_bytes(duration_sec=3.0)
    files = {"file": ("test_wave.wav", wav, "audio/wav")}
    res = client.post("/api/waveform", files=files, data={"points": 50})
    assert res.status_code == 200
    data = res.json()
    assert len(data["peaks"]) == 50
    assert all(0.0 <= p <= 1.0 for p in data["peaks"])


def test_history_lifecycle(client):
    # Clear history
    del_res = client.delete("/api/history")
    assert del_res.status_code == 200

    # Submit an analysis
    wav = _make_wav_bytes(duration_sec=2.0)
    client.post("/api/analyze", files={"file": ("history_test.wav", wav, "audio/wav")})

    # Retrieve history
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) >= 1
    assert history[0]["filename"] == "history_test.wav"
