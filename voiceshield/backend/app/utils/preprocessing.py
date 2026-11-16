"""
VoiceShield — Audio Preprocessing Utilities
============================================
Handles audio standardization, sample rate conversion, channel downmixing,
normalization, sliding-window chunking, and waveform visualization downsampling.
"""

import logging
from typing import List, Tuple
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

# AASIST expects 64,600 samples at 16kHz (approx 4.0375s)
AASIST_NB_SAMP = 64600
TARGET_SAMPLE_RATE = 16000


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to single-channel mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=-1 if audio.shape[-1] <= 8 else 0)


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1.0, 1.0]."""
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        return audio / peak
    return audio


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample 1D audio array to target sample rate using torchaudio."""
    if orig_sr == target_sr:
        return audio

    tensor = torch.from_numpy(audio).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(tensor).squeeze(0).numpy()
    return resampled


def pad_or_crop(audio: np.ndarray, target_samples: int = AASIST_NB_SAMP) -> np.ndarray:
    """
    Pad (via tiling) or crop audio to exact target_samples length.
    Follows official AASIST evaluation data protocol.
    """
    length = len(audio)
    if length >= target_samples:
        return audio[:target_samples]

    if length == 0:
        return np.zeros(target_samples, dtype=np.float32)

    # Tile repeated pattern to reach target length
    repeats = int(np.ceil(target_samples / length))
    tiled = np.tile(audio, repeats)[:target_samples]
    return tiled


def slice_windows(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    window_samples: int = AASIST_NB_SAMP,
    hop_samples: int = int(TARGET_SAMPLE_RATE * 2.0)
) -> List[Tuple[int, float, float, np.ndarray]]:
    """
    Slice audio into overlapping windows for segment-level deepfake analysis.

    Returns:
        List of tuples: (window_index, start_sec, end_sec, window_audio_array)
    """
    total_samples = len(audio)
    if total_samples <= window_samples:
        padded = pad_or_crop(audio, window_samples)
        return [(0, 0.0, total_samples / sample_rate, padded)]

    windows = []
    idx = 0
    start = 0

    while start + window_samples <= total_samples:
        chunk = audio[start : start + window_samples]
        start_sec = start / sample_rate
        end_sec = (start + window_samples) / sample_rate
        windows.append((idx, round(start_sec, 2), round(end_sec, 2), chunk))
        start += hop_samples
        idx += 1

    # If the remaining tail is at least 1.0 second, include a final tail window
    if total_samples - start >= sample_rate:
        tail = audio[-window_samples:]
        start_sec = max(0.0, (total_samples - window_samples) / sample_rate)
        end_sec = total_samples / sample_rate
        windows.append((idx, round(start_sec, 2), round(end_sec, 2), tail))

    return windows


def generate_waveform_peaks(audio: np.ndarray, num_points: int = 120) -> List[float]:
    """
    Downsample audio to `num_points` peak amplitude values for frontend waveform visualization.
    Normalized to range [0.0, 1.0].
    """
    if len(audio) == 0:
        return [0.0] * num_points

    chunk_size = max(1, len(audio) // num_points)
    peaks = []
    for i in range(num_points):
        start = i * chunk_size
        end = min(len(audio), start + chunk_size)
        if start >= len(audio):
            peaks.append(0.0)
        else:
            chunk = audio[start:end]
            val = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
            peaks.append(round(min(1.0, val), 3))

    return peaks
