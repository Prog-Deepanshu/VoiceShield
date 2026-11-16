"""
VoiceGuard — Audio Stream Buffer
=================================
Thread-safe ring buffer for accumulating streaming PCM audio chunks
from WebSocket connections until enough data is available for analysis.
"""

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioStreamBuffer:
    """
    Thread-safe audio accumulation buffer for real-time streaming.

    Accepts raw PCM audio chunks (16-bit signed integer or float32),
    accumulates them into a float32 numpy array, and signals readiness
    once the target duration is reached.

    Usage:
        buffer = AudioStreamBuffer(sample_rate=16000, target_duration_sec=6.0)
        buffer.append(pcm_bytes_chunk_1)
        buffer.append(pcm_bytes_chunk_2)
        if buffer.is_ready():
            audio = buffer.get_audio()
            buffer.reset()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_duration_sec: float = 6.0,
        dtype: str = "int16",
    ):
        """
        Initialize the audio buffer.

        Args:
            sample_rate:       Audio sample rate in Hz
            target_duration_sec: Target buffering duration in seconds
            dtype:             Input PCM encoding: 'int16' or 'float32'
        """
        self.sample_rate = sample_rate
        self.target_duration_sec = target_duration_sec
        self.target_samples = int(sample_rate * target_duration_sec)
        self.dtype = dtype

        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._total_samples: int = 0

    def append(self, chunk: bytes) -> None:
        """
        Append raw PCM audio bytes to the buffer.

        Args:
            chunk: Raw PCM audio data (little-endian)
        """
        if self.dtype == "int16":
            audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        elif self.dtype == "float32":
            audio = np.frombuffer(chunk, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        with self._lock:
            self._chunks.append(audio)
            self._total_samples += len(audio)

    def append_array(self, audio: np.ndarray) -> None:
        """
        Append a numpy float32 audio array directly.

        Args:
            audio: Float32 audio samples, expected range [-1, 1]
        """
        audio = audio.astype(np.float32)
        with self._lock:
            self._chunks.append(audio)
            self._total_samples += len(audio)

    def is_ready(self) -> bool:
        """Check if enough audio has been buffered for analysis."""
        with self._lock:
            return self._total_samples >= self.target_samples

    @property
    def buffered_seconds(self) -> float:
        """Current amount of buffered audio in seconds."""
        with self._lock:
            return self._total_samples / self.sample_rate

    @property
    def buffered_samples(self) -> int:
        """Current number of buffered samples."""
        with self._lock:
            return self._total_samples

    def get_audio(self, max_samples: Optional[int] = None) -> np.ndarray:
        """
        Get the accumulated audio as a single float32 array.

        If more audio is buffered than needed, returns the most recent
        `max_samples` samples (tail). If less, returns all available.

        Args:
            max_samples: Maximum samples to return (None = all)

        Returns:
            Float32 numpy array of audio samples
        """
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)

            audio = np.concatenate(self._chunks)

            if max_samples is not None and len(audio) > max_samples:
                audio = audio[-max_samples:]

            return audio

    def reset(self) -> None:
        """Clear the buffer for the next analysis window."""
        with self._lock:
            self._chunks.clear()
            self._total_samples = 0

    def __len__(self) -> int:
        with self._lock:
            return self._total_samples

    def __repr__(self) -> str:
        return (
            f"AudioStreamBuffer("
            f"buffered={self.buffered_seconds:.2f}s / "
            f"{self.target_duration_sec:.1f}s, "
            f"samples={self.buffered_samples}/{self.target_samples})"
        )
