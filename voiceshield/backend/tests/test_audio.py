"""
Unit tests for audio processing and diagnostic utilities.
"""

import io
import numpy as np
import soundfile as sf
import pytest

from app.services.audio_service import AudioService
from app.utils.preprocessing import (
    normalize_peak,
    pad_or_crop,
    slice_windows,
    to_mono,
    AASIST_NB_SAMP,
    TARGET_SAMPLE_RATE,
)


def _generate_sine_wav(duration_sec: float = 2.0, sr: int = 16000, freq: float = 440.0) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, sine, sr, format="WAV")
    return bio.getvalue()


def test_to_mono():
    stereo = np.ones((100, 2), dtype=np.float32)
    mono = to_mono(stereo)
    assert mono.ndim == 1
    assert len(mono) == 100


def test_normalize_peak():
    signal = np.array([0.2, -0.8, 0.4], dtype=np.float32)
    norm = normalize_peak(signal)
    assert np.max(np.abs(norm)) == pytest.approx(1.0, 1e-5)


def test_pad_or_crop():
    short_signal = np.ones(1000, dtype=np.float32)
    padded = pad_or_crop(short_signal, AASIST_NB_SAMP)
    assert len(padded) == AASIST_NB_SAMP

    long_signal = np.ones(100000, dtype=np.float32)
    cropped = pad_or_crop(long_signal, AASIST_NB_SAMP)
    assert len(cropped) == AASIST_NB_SAMP


def test_slice_windows():
    sr = 16000
    # 10 second audio
    audio = np.random.randn(sr * 10).astype(np.float32)
    windows = slice_windows(audio, sample_rate=sr)
    assert len(windows) > 1
    for idx, start_sec, end_sec, chunk in windows:
        assert len(chunk) == AASIST_NB_SAMP
        assert end_sec > start_sec


def test_audio_service_decode():
    wav_bytes = _generate_sine_wav(duration_sec=3.0, sr=16000)
    audio, sr, meta = AudioService.decode_audio_bytes(wav_bytes, "sine.wav")
    assert sr == 16000
    assert meta.duration_sec == pytest.approx(3.0, 0.1)
    assert meta.channels == 1
    assert len(audio) == 16000 * 3


def test_spectral_diagnostics():
    wav_bytes = _generate_sine_wav(duration_sec=2.0, sr=16000, freq=220.0)
    audio, sr, _ = AudioService.decode_audio_bytes(wav_bytes, "sine.wav")
    diag = AudioService.extract_diagnostics(audio, sr)
    assert diag.energy_rms > 0
    assert diag.pitch_mean_hz > 0
    assert diag.spectral_centroid_hz > 0
