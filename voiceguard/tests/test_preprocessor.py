"""
VoiceGuard — Preprocessor Tests
==================================
"""

import numpy as np
import pytest

from core.preprocessor import AudioPreprocessor, AASIST_NB_SAMP


class TestAudioPreprocessor:
    """Test the AudioPreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        return AudioPreprocessor(
            target_sr=16000,
            nb_samp=AASIST_NB_SAMP,
            enable_noise_reduction=False,  # Disable for speed in tests
        )

    def test_output_shape(self, preprocessor):
        """Output tensor should have exactly nb_samp samples."""
        audio = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds
        result = preprocessor.preprocess(audio, sample_rate=16000)
        assert result.shape == (AASIST_NB_SAMP,)

    def test_short_audio_padding(self, preprocessor):
        """Short audio should be padded via tiling."""
        audio = np.random.randn(8000).astype(np.float32)  # 0.5 seconds
        result = preprocessor.preprocess(audio, sample_rate=16000)
        assert result.shape == (AASIST_NB_SAMP,)
        assert not np.all(result.numpy() == 0)  # Should not be zero-padded

    def test_long_audio_cropping(self, preprocessor):
        """Long audio should be cropped to nb_samp."""
        audio = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds
        result = preprocessor.preprocess(audio, sample_rate=16000)
        assert result.shape == (AASIST_NB_SAMP,)

    def test_normalization(self, preprocessor):
        """Output should be normalized to [-1, 1] range."""
        audio = np.random.randn(16000 * 3).astype(np.float32) * 5.0
        result = preprocessor.preprocess(audio, sample_rate=16000)
        assert result.max().item() <= 1.0
        assert result.min().item() >= -1.0

    def test_mono_conversion(self, preprocessor):
        """Multi-channel audio should be converted to mono."""
        audio = np.random.randn(16000 * 3, 2).astype(np.float32)  # Stereo
        result = preprocessor.preprocess(audio, sample_rate=16000)
        assert result.shape == (AASIST_NB_SAMP,)

    def test_resampling(self, preprocessor):
        """Audio at different sample rates should be resampled to 16kHz."""
        # 8kHz audio (2 seconds)
        audio = np.random.randn(8000 * 2).astype(np.float32)
        result = preprocessor.preprocess(audio, sample_rate=8000)
        assert result.shape == (AASIST_NB_SAMP,)

    def test_pad_to_length_exact(self):
        """Audio of exact length should pass through unchanged."""
        audio = np.random.randn(AASIST_NB_SAMP).astype(np.float32)
        result = AudioPreprocessor._pad_to_length(audio, AASIST_NB_SAMP)
        assert len(result) == AASIST_NB_SAMP
        np.testing.assert_array_equal(result, audio)

    def test_silence_removal(self, preprocessor):
        """Leading/trailing silence should be trimmed."""
        # Create audio with silence at start and end
        audio = np.zeros(16000 * 4, dtype=np.float32)
        audio[16000:48000] = np.random.randn(32000).astype(np.float32) * 0.5
        result = preprocessor.preprocess(audio, sample_rate=16000)
        assert result.shape == (AASIST_NB_SAMP,)
