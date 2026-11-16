"""
VoiceGuard — Audio Buffer Tests
=================================
"""

import numpy as np
import pytest

from core.audio_buffer import AudioStreamBuffer


class TestAudioStreamBuffer:
    """Test the AudioStreamBuffer class."""

    def test_initialization(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=6.0)
        assert buffer.sample_rate == 16000
        assert buffer.target_samples == 96000
        assert buffer.buffered_samples == 0
        assert not buffer.is_ready()

    def test_append_int16(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=1.0, dtype="int16")
        # Create 0.5s of int16 PCM audio
        audio = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        buffer.append(audio.tobytes())
        assert buffer.buffered_samples == 8000
        assert abs(buffer.buffered_seconds - 0.5) < 0.01

    def test_append_float32(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=1.0, dtype="float32")
        audio = np.random.randn(8000).astype(np.float32)
        buffer.append(audio.tobytes())
        assert buffer.buffered_samples == 8000

    def test_append_array(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=1.0)
        audio = np.random.randn(8000).astype(np.float32)
        buffer.append_array(audio)
        assert buffer.buffered_samples == 8000

    def test_is_ready(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=1.0)
        assert not buffer.is_ready()

        # Add 0.5s (not enough)
        buffer.append_array(np.zeros(8000, dtype=np.float32))
        assert not buffer.is_ready()

        # Add another 0.5s (now enough)
        buffer.append_array(np.zeros(8000, dtype=np.float32))
        assert buffer.is_ready()

    def test_get_audio(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=1.0)
        audio1 = np.ones(8000, dtype=np.float32) * 0.5
        audio2 = np.ones(8000, dtype=np.float32) * -0.5
        buffer.append_array(audio1)
        buffer.append_array(audio2)

        result = buffer.get_audio()
        assert len(result) == 16000
        assert np.allclose(result[:8000], 0.5)
        assert np.allclose(result[8000:], -0.5)

    def test_get_audio_with_max_samples(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=2.0)
        buffer.append_array(np.random.randn(32000).astype(np.float32))

        result = buffer.get_audio(max_samples=16000)
        assert len(result) == 16000  # Should only return last 16000 samples

    def test_reset(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=1.0)
        buffer.append_array(np.zeros(16000, dtype=np.float32))
        assert buffer.is_ready()

        buffer.reset()
        assert not buffer.is_ready()
        assert buffer.buffered_samples == 0

    def test_empty_get_audio(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=1.0)
        result = buffer.get_audio()
        assert len(result) == 0

    def test_repr(self):
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=6.0)
        repr_str = repr(buffer)
        assert "AudioStreamBuffer" in repr_str
        assert "0.00s" in repr_str
