"""
VoiceShield — Audio Service
===========================
Service responsible for audio file ingestion, decoding across various formats,
metadata extraction, waveform summary generation, and spectral diagnostics.
"""

import io
import logging
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

from app.models.schemas import AudioMetadata, SpectralDiagnostics
from app.utils.preprocessing import (
    normalize_peak,
    resample_audio,
    to_mono,
    generate_waveform_peaks,
    TARGET_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class AudioService:
    """Service handling audio file decoding and acoustic diagnostics."""

    @staticmethod
    def decode_audio_bytes(
        file_bytes: bytes,
        filename: str,
        fallback_sr: int = 16000
    ) -> Tuple[np.ndarray, int, AudioMetadata]:
        """
        Safely decode audio bytes into a float32 numpy array and metadata.
        Supports WAV, MP3, FLAC, OGG, and raw PCM.
        """
        file_size = len(file_bytes)
        format_detected = "unknown"

        try:
            with io.BytesIO(file_bytes) as bio:
                audio_data, sr = sf.read(bio, dtype="float32")
                # Detect format from soundfile info
                bio.seek(0)
                info = sf.info(bio)
                format_detected = info.format.lower()
                channels = info.channels
                duration = len(audio_data) / sr

        except Exception as sf_err:
            logger.warning(f"SoundFile decode failed for {filename} ({sf_err}). Falling back to raw PCM 16-bit.")
            try:
                # Fallback: assume raw 16-bit PCM little-endian
                raw_int16 = np.frombuffer(file_bytes, dtype=np.int16)
                audio_data = raw_int16.astype(np.float32) / 32768.0
                sr = fallback_sr
                channels = 1
                duration = len(audio_data) / sr
                format_detected = "raw_pcm"
            except Exception as pcm_err:
                raise ValueError(f"Could not decode audio stream: {sf_err}; fallback PCM error: {pcm_err}")

        # Downmix to mono if multi-channel
        mono_audio = to_mono(audio_data)
        normalized_audio = normalize_peak(mono_audio)

        metadata = AudioMetadata(
            filename=filename,
            duration_sec=round(duration, 3),
            sample_rate=sr,
            channels=channels,
            format=format_detected,
            file_size_bytes=file_size
        )

        return normalized_audio, sr, metadata

    @staticmethod
    def extract_diagnostics(audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> SpectralDiagnostics:
        """
        Extract acoustic features correlating with synthetic speech artifacts:
        - Pitch stability (TTS synthesizers often exhibit abnormally smooth or unnatural pitch)
        - Spectral flatness (Neural vocoders introduce flat noise floors)
        - Harmonic ratio & Rolloff
        """
        try:
            # Resample to 16kHz for uniform spectral analysis if different
            if sr != TARGET_SAMPLE_RATE:
                audio = resample_audio(audio, sr, TARGET_SAMPLE_RATE)
                sr = TARGET_SAMPLE_RATE

            # RMS Energy
            energy_rms = float(np.sqrt(np.mean(audio ** 2)))

            # FFT Spectrum
            n_fft = 2048
            if len(audio) < n_fft:
                spectrum = np.fft.rfft(audio, n=n_fft)
            else:
                spectrum = np.fft.rfft(audio[:n_fft * 4], n=n_fft)

            mag = np.abs(spectrum)
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

            # Spectral Centroid (Center of mass)
            mag_sum = np.sum(mag)
            centroid = float(np.sum(freqs * mag) / (mag_sum + 1e-12))

            # Spectral Rolloff (85% energy threshold)
            cum_energy = np.cumsum(mag)
            threshold = 0.85 * cum_energy[-1] if len(cum_energy) > 0 else 0
            idx_rolloff = np.where(cum_energy >= threshold)[0]
            rolloff = float(freqs[idx_rolloff[0]]) if len(idx_rolloff) > 0 else 0.0

            # Spectral Flatness (Geometric mean / Arithmetic mean of power spectrum)
            power = mag ** 2 + 1e-12
            geom_mean = np.exp(np.mean(np.log(power)))
            arith_mean = np.mean(power)
            flatness = float(geom_mean / arith_mean)

            # Pitch estimation via Autocorrelation
            corr = np.correlate(audio[:sr * 2] if len(audio) >= sr * 2 else audio, audio[:sr * 2] if len(audio) >= sr * 2 else audio, mode="full")
            corr = corr[len(corr) // 2 :]

            # Search in valid human pitch range: 60Hz to 400Hz
            min_lag = int(sr / 400)
            max_lag = int(sr / 60)
            if len(corr) > max_lag and max_lag > min_lag:
                peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                pitch_hz = float(sr / peak_lag) if peak_lag > 0 else 140.0
                harmonic_ratio = float(corr[peak_lag] / (corr[0] + 1e-12))
            else:
                pitch_hz = 140.0
                harmonic_ratio = 1.0

            # Pitch stability index (standard deviation across short frames)
            frame_len = int(sr * 0.05)
            n_frames = min(20, max(1, len(audio) // frame_len))
            frame_pitches = []
            for i in range(n_frames):
                frame = audio[i * frame_len : (i + 1) * frame_len]
                if len(frame) > min_lag:
                    c = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
                    if len(c) > max_lag:
                        p_lag = min_lag + np.argmax(c[min_lag:max_lag])
                        if p_lag > 0:
                            frame_pitches.append(sr / p_lag)

            pitch_std = float(np.std(frame_pitches)) if len(frame_pitches) > 1 else 5.0
            pitch_stability = round(float(pitch_std / (pitch_hz + 1e-6)), 4)

            return SpectralDiagnostics(
                pitch_stability=pitch_stability,
                pitch_mean_hz=round(pitch_hz, 1),
                spectral_flatness=round(flatness, 6),
                spectral_centroid_hz=round(centroid, 1),
                spectral_rolloff_hz=round(rolloff, 1),
                harmonic_ratio=round(max(0.0, harmonic_ratio), 4),
                energy_rms=round(energy_rms, 6)
            )

        except Exception as e:
            logger.warning(f"Error computing spectral diagnostics: {e}")
            return SpectralDiagnostics(
                pitch_stability=0.05,
                pitch_mean_hz=140.0,
                spectral_flatness=0.001,
                spectral_centroid_hz=1800.0,
                spectral_rolloff_hz=3200.0,
                harmonic_ratio=1.5,
                energy_rms=0.02
            )

    @staticmethod
    def get_waveform_preview(audio: np.ndarray, num_points: int = 120) -> list[float]:
        """Generate normalized amplitude peaks for UI rendering."""
        return generate_waveform_peaks(audio, num_points)
