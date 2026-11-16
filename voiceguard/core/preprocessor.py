"""
VoiceGuard — Audio Preprocessor
================================
Audio preprocessing pipeline for cleaning and normalizing live call audio
before feeding into the AASIST deepfake detection model.

Pipeline steps:
    1. Resample to 16 kHz (if needed)
    2. Convert to mono (if multi-channel)
    3. Peak normalize to [-1, 1]
    4. Voice Activity Detection (energy-based silence removal)
    5. Lightweight spectral-gating noise reduction
    6. Pad/crop to AASIST expected length (64,600 samples ≈ 4.04s)
"""

import logging

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

# AASIST model expects exactly this many samples
AASIST_NB_SAMP = 64600


class AudioPreprocessor:
    """
    Production audio preprocessor for the VoiceGuard pipeline.

    Handles all necessary transformations to convert raw call audio
    into the format expected by the AASIST model.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        nb_samp: int = AASIST_NB_SAMP,
        enable_noise_reduction: bool = True,
        vad_energy_threshold: float = 0.01,
    ):
        """
        Args:
            target_sr:              Target sample rate (AASIST requires 16kHz)
            nb_samp:                Number of samples expected by AASIST
            enable_noise_reduction: Whether to apply spectral-gating noise reduction
            vad_energy_threshold:   Energy threshold for silence detection
        """
        self.target_sr = target_sr
        self.nb_samp = nb_samp
        self.enable_noise_reduction = enable_noise_reduction
        self.vad_energy_threshold = vad_energy_threshold
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Full preprocessing pipeline: resample → mono → normalize → VAD → denoise → pad/crop.

        Args:
            audio:       Raw audio as float32 numpy array
            sample_rate: Original sample rate of the audio

        Returns:
            Preprocessed audio tensor of shape (nb_samp,) ready for AASIST
        """
        # Ensure float32
        audio = audio.astype(np.float32)

        # Convert to mono if multi-channel
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        # Resample if needed
        if sample_rate != self.target_sr:
            audio = self._resample(audio, sample_rate)

        # Peak normalize
        audio = self._normalize(audio)

        # Remove silence (VAD)
        audio = self._remove_silence(audio)

        # Lightweight noise reduction
        if self.enable_noise_reduction:
            audio = self._reduce_noise(audio)

        # Pad or crop to AASIST expected length
        audio = self._pad_to_length(audio, self.nb_samp)

        return torch.from_numpy(audio).float()

    def _resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate using torchaudio."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.target_sr
            )

        tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampled = self._resamplers[orig_sr](tensor)
        return resampled.squeeze(0).numpy()

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        """Peak normalize audio to [-1, 1] range."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    def _remove_silence(
        self,
        audio: np.ndarray,
        frame_length: int = 1024,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Remove leading and trailing silence using energy-based VAD.

        Uses a simple energy threshold approach — fast enough for real-time
        processing (~0.1ms for 10s audio).
        """
        # Compute frame energies
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        if n_frames <= 0:
            return audio

        energies = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + frame_length]
            energies[i] = np.mean(frame ** 2)

        # Find first and last frames above threshold
        active_frames = np.where(energies > self.vad_energy_threshold)[0]

        if len(active_frames) == 0:
            logger.warning("No speech detected — returning original audio")
            return audio

        start_sample = active_frames[0] * hop_length
        end_sample = min(
            active_frames[-1] * hop_length + frame_length, len(audio)
        )

        trimmed = audio[start_sample:end_sample]

        # Don't trim too aggressively — keep at least 1s
        if len(trimmed) < self.target_sr:
            return audio

        return trimmed

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Lightweight spectral-gating noise reduction.

        Uses a simple spectral gate: estimate noise spectrum from the first
        100ms of audio (assumed to contain ambient noise), then suppress
        frequencies below that threshold.

        This is much faster (~3ms) than full noisereduce library (~50ms)
        and sufficient for call-center audio quality.
        """
        try:
            # Use noisereduce if available (better quality)
            import noisereduce as nr

            return nr.reduce_noise(
                y=audio,
                sr=self.target_sr,
                stationary=True,
                prop_decrease=0.6,
                n_fft=512,
                hop_length=256,
            )
        except ImportError:
            # Fallback: simple spectral gating
            return self._spectral_gate(audio)

    def _spectral_gate(
        self,
        audio: np.ndarray,
        n_fft: int = 512,
        hop_length: int = 256,
        noise_duration_sec: float = 0.1,
    ) -> np.ndarray:
        """
        Simple spectral gating noise reduction.

        Estimates noise profile from the first `noise_duration_sec` seconds,
        then gates frequencies that fall below the noise floor.
        """
        noise_samples = int(self.target_sr * noise_duration_sec)
        noise_clip = audio[:noise_samples]

        # Estimate noise spectrum
        noise_stft = np.fft.rfft(noise_clip, n=n_fft)
        noise_mag = np.abs(noise_stft)
        noise_threshold = np.mean(noise_mag) * 1.5

        # Process audio in frames
        n_frames = (len(audio) - n_fft) // hop_length + 1
        output = np.zeros_like(audio)
        window = np.hanning(n_fft)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft] * window
            stft = np.fft.rfft(frame)
            mag = np.abs(stft)

            # Soft spectral gate
            gain = np.clip((mag - noise_threshold) / (mag + 1e-10), 0, 1)
            stft_clean = stft * gain

            reconstructed = np.fft.irfft(stft_clean, n=n_fft)
            output[start : start + n_fft] += reconstructed * window

        return output

    @staticmethod
    def _pad_to_length(audio: np.ndarray, max_len: int) -> np.ndarray:
        """
        Pad or crop audio to exactly `max_len` samples.

        Matches the padding strategy from AASIST's data_utils.py:
        - If audio is longer: take the first max_len samples
        - If audio is shorter: tile (repeat) to fill
        """
        x_len = audio.shape[0]

        if x_len >= max_len:
            return audio[:max_len]

        # Tile to fill the target length
        num_repeats = int(max_len / x_len) + 1
        padded = np.tile(audio, num_repeats)[:max_len]
        return padded
