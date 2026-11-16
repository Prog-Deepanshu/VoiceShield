"""
VoiceGuard — Feature Extractor
===============================
Supplementary feature extraction for diagnostic logging, visualization,
and spectral analysis. These features are NOT fed to the AASIST model
(which operates on raw waveform), but are useful for:

    - Audit logging of spectral characteristics
    - Identifying specific artifact types (pitch, harmonics, vocoder)
    - Visualization in dashboards
    - Secondary heuristic checks
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpectralAnalysis:
    """Container for spectral diagnostic features."""

    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_flatness: float = 0.0
    zero_crossing_rate: float = 0.0
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_stability: float = 0.0  # Lower = more stable (human-like)
    harmonic_ratio: float = 0.0
    energy_rms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "spectral_centroid_hz": round(self.spectral_centroid, 2),
            "spectral_bandwidth_hz": round(self.spectral_bandwidth, 2),
            "spectral_rolloff_hz": round(self.spectral_rolloff, 2),
            "spectral_flatness": round(self.spectral_flatness, 6),
            "zero_crossing_rate": round(self.zero_crossing_rate, 6),
            "pitch_mean_hz": round(self.pitch_mean, 2),
            "pitch_std_hz": round(self.pitch_std, 2),
            "pitch_stability": round(self.pitch_stability, 4),
            "harmonic_ratio": round(self.harmonic_ratio, 4),
            "energy_rms": round(self.energy_rms, 6),
        }


class FeatureExtractor:
    """
    Diagnostic feature extractor for audio analysis reporting.

    Extracts spectral and temporal features that help explain
    *why* audio was flagged as synthetic. These features correlate
    with common deepfake artifacts:

    - Pitch instability → TTS models often have unnaturally smooth pitch
    - Spectral flatness → vocoder artifacts produce flatter spectra
    - Harmonic ratio    → synthetic speech may have weaker harmonics
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def analyze(self, audio: np.ndarray) -> SpectralAnalysis:
        """
        Run full diagnostic analysis on audio.

        Args:
            audio: Float32 audio array, expected range [-1, 1]

        Returns:
            SpectralAnalysis dataclass with computed features
        """
        analysis = SpectralAnalysis()

        try:
            analysis.energy_rms = self._compute_rms(audio)
            analysis.zero_crossing_rate = self._compute_zcr(audio)

            # Spectral features
            magnitude, freqs = self._compute_spectrum(audio)
            analysis.spectral_centroid = self._spectral_centroid(magnitude, freqs)
            analysis.spectral_bandwidth = self._spectral_bandwidth(
                magnitude, freqs, analysis.spectral_centroid
            )
            analysis.spectral_rolloff = self._spectral_rolloff(magnitude, freqs)
            analysis.spectral_flatness = self._spectral_flatness(magnitude)

            # Pitch analysis
            pitch_contour = self._estimate_pitch(audio)
            if len(pitch_contour) > 0:
                analysis.pitch_mean = float(np.mean(pitch_contour))
                analysis.pitch_std = float(np.std(pitch_contour))
                analysis.pitch_stability = (
                    analysis.pitch_std / (analysis.pitch_mean + 1e-10)
                )

            # Harmonic ratio
            analysis.harmonic_ratio = self._harmonic_ratio(audio)

        except Exception as e:
            logger.warning(f"Feature extraction partially failed: {e}")

        return analysis

    @staticmethod
    def _compute_rms(audio: np.ndarray) -> float:
        """Root Mean Square energy."""
        return float(np.sqrt(np.mean(audio ** 2)))

    @staticmethod
    def _compute_zcr(audio: np.ndarray) -> float:
        """Zero Crossing Rate — average rate of sign changes."""
        signs = np.sign(audio)
        crossings = np.abs(np.diff(signs))
        return float(np.mean(crossings > 0))

    def _compute_spectrum(
        self, audio: np.ndarray, n_fft: int = 2048
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute magnitude spectrum and frequency bins."""
        spectrum = np.fft.rfft(audio, n=n_fft)
        magnitude = np.abs(spectrum)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
        return magnitude, freqs

    @staticmethod
    def _spectral_centroid(magnitude: np.ndarray, freqs: np.ndarray) -> float:
        """Weighted mean of frequencies — 'center of mass' of spectrum."""
        mag_sum = np.sum(magnitude)
        if mag_sum == 0:
            return 0.0
        return float(np.sum(freqs * magnitude) / mag_sum)

    @staticmethod
    def _spectral_bandwidth(
        magnitude: np.ndarray, freqs: np.ndarray, centroid: float
    ) -> float:
        """Weighted standard deviation of frequencies around centroid."""
        mag_sum = np.sum(magnitude)
        if mag_sum == 0:
            return 0.0
        return float(
            np.sqrt(np.sum(magnitude * (freqs - centroid) ** 2) / mag_sum)
        )

    @staticmethod
    def _spectral_rolloff(
        magnitude: np.ndarray, freqs: np.ndarray, percentile: float = 0.85
    ) -> float:
        """Frequency below which `percentile` of spectral energy is contained."""
        cumulative = np.cumsum(magnitude)
        total = cumulative[-1]
        if total == 0:
            return 0.0
        rolloff_idx = np.searchsorted(cumulative, percentile * total)
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        return float(freqs[rolloff_idx])

    @staticmethod
    def _spectral_flatness(magnitude: np.ndarray) -> float:
        """
        Ratio of geometric mean to arithmetic mean of spectrum.
        Values close to 1.0 indicate noise-like (flat) spectrum.
        Values close to 0.0 indicate tonal (peaked) spectrum.
        Vocoder artifacts often produce higher flatness.
        """
        magnitude = magnitude + 1e-10  # avoid log(0)
        geo_mean = np.exp(np.mean(np.log(magnitude)))
        arith_mean = np.mean(magnitude)
        return float(geo_mean / arith_mean)

    def _estimate_pitch(
        self,
        audio: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512,
        fmin: float = 50.0,
        fmax: float = 500.0,
    ) -> np.ndarray:
        """
        Estimate fundamental frequency (F0) contour using autocorrelation.

        Returns array of F0 values in Hz for each voiced frame.
        """
        min_lag = int(self.sample_rate / fmax)
        max_lag = int(self.sample_rate / fmin)

        n_frames = (len(audio) - frame_length) // hop_length + 1
        pitches = []

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + frame_length]

            # Check if frame has enough energy (voiced)
            energy = np.mean(frame ** 2)
            if energy < 0.001:
                continue

            # Autocorrelation-based pitch detection
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]  # keep positive lags

            # Search for peak in valid lag range
            if max_lag >= len(corr):
                continue

            search_region = corr[min_lag : max_lag + 1]
            if len(search_region) == 0:
                continue

            peak_idx = np.argmax(search_region) + min_lag

            if corr[peak_idx] > 0.3 * corr[0]:  # confidence threshold
                pitch = self.sample_rate / peak_idx
                pitches.append(pitch)

        return np.array(pitches)

    def _harmonic_ratio(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
    ) -> float:
        """
        Estimate harmonic-to-noise ratio.
        Higher values indicate more harmonic (speech-like) content.
        Synthetic speech may show different harmonic patterns.
        """
        spectrum = np.fft.rfft(audio, n=n_fft)
        magnitude = np.abs(spectrum) ** 2

        # Simple harmonic detection via cepstral analysis
        log_spectrum = np.log(magnitude + 1e-10)
        cepstrum = np.fft.irfft(log_spectrum)

        # Harmonic peak is in the quefrency range corresponding to F0
        min_quefrency = int(self.sample_rate / 500)  # max F0 = 500 Hz
        max_quefrency = int(self.sample_rate / 50)   # min F0 = 50 Hz

        if max_quefrency >= len(cepstrum):
            return 0.0

        peak_region = np.abs(cepstrum[min_quefrency:max_quefrency])
        if len(peak_region) == 0:
            return 0.0

        harmonic_peak = np.max(peak_region)
        noise_floor = np.mean(np.abs(cepstrum[:min_quefrency]))

        if noise_floor == 0:
            return 0.0

        return float(harmonic_peak / (noise_floor + 1e-10))
