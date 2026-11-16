"""
VoiceShield — Real-Time Call Protection Framework
===================================================
Production abstraction layer designed for live telecommunication call protection.
Decouples audio transport (WebSocket, SIP/RTP, WebRTC) from detection engines.

Architecture:
    AudioStreamSource (WebSocket / VoIP / RTP / Recording)
            │
            ▼
    AudioBufferManager (Ring buffer, configurable windows & overlap)
            │
            ▼
    VoiceDetectionEngine (VoiceGuard AASIST / Ensemble)
            │
            ▼
    DetectionEventHandler (Temporal smoothing, alerting, event dispatch)
"""

import abc
import asyncio
import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from app.models.schemas import ClassificationState

logger = logging.getLogger(__name__)


# ── 1. AudioStreamSource Abstraction ─────────────────────────────────

class AudioStreamSource(abc.ABC):
    """
    Abstract interface for streaming audio sources.
    Can be backed by WebSockets, SIP/RTP RTP streams, WebRTC tracks, or test feeds.
    """

    @abc.abstractmethod
    async def read_chunk(self) -> Optional[bytes]:
        """Read the next chunk of raw PCM audio bytes. Returns None when stream ends."""
        pass

    @abc.abstractmethod
    def get_source_info(self) -> Dict[str, str]:
        """Return diagnostic metadata about the audio transport."""
        pass


class WebSocketStreamSource(AudioStreamSource):
    """Audio source reading from a live WebSocket connection."""

    def __init__(self, queue: asyncio.Queue, call_id: str, sample_rate: int = 16000):
        self.queue = queue
        self.call_id = call_id
        self.sample_rate = sample_rate

    async def read_chunk(self) -> Optional[bytes]:
        return await self.queue.get()

    def get_source_info(self) -> Dict[str, str]:
        return {
            "type": "websocket",
            "call_id": self.call_id,
            "sample_rate": str(self.sample_rate)
        }


# ── 2. AudioBufferManager ─────────────────────────────────────────────

class AudioBufferManager:
    """
    Thread-safe accumulating ring buffer for streaming PCM audio chunks.
    Maintains configurable analysis windows with context overlap between successive evaluations.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_duration_sec: float = 6.0,
        overlap_duration_sec: float = 2.0,
        dtype: str = "int16"
    ):
        self.sample_rate = sample_rate
        self.window_duration_sec = window_duration_sec
        self.overlap_duration_sec = overlap_duration_sec
        self.window_samples = int(sample_rate * window_duration_sec)
        self.overlap_samples = int(sample_rate * overlap_duration_sec)
        self.dtype = dtype

        self._buffer: List[np.ndarray] = []
        self._total_samples: int = 0

    def append_pcm_bytes(self, pcm_bytes: bytes) -> None:
        """Decode and append raw PCM bytes (int16 little-endian or float32)."""
        if len(pcm_bytes) == 0:
            return

        if self.dtype == "int16":
            arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif self.dtype == "float32":
            arr = np.frombuffer(pcm_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        self._buffer.append(arr)
        self._total_samples += len(arr)

    def append_array(self, audio: np.ndarray) -> None:
        """Append float32 numpy array directly."""
        audio = audio.astype(np.float32)
        self._buffer.append(audio)
        self._total_samples += len(audio)

    def is_window_ready(self) -> bool:
        """Check if enough samples have accumulated to run an inference window."""
        return self._total_samples >= self.window_samples

    @property
    def buffered_seconds(self) -> float:
        return self._total_samples / self.sample_rate

    def extract_window(self) -> np.ndarray:
        """
        Concatenate buffered audio and return an analysis window.
        Keeps the trailing overlap samples in the buffer for continuity.
        """
        if not self._buffer:
            return np.array([], dtype=np.float32)

        full_audio = np.concatenate(self._buffer)

        # Slice the analysis window from the beginning
        window = full_audio[: self.window_samples]

        # Retain trailing overlap
        if len(full_audio) > self.window_samples:
            remaining = full_audio[self.window_samples - self.overlap_samples :]
        else:
            remaining = full_audio[-self.overlap_samples :] if len(full_audio) >= self.overlap_samples else full_audio

        self._buffer = [remaining]
        self._total_samples = len(remaining)

        return window

    def flush_remaining(self) -> np.ndarray:
        """Drain all remaining audio when stream ends."""
        if not self._buffer:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(self._buffer)
        self.reset()
        return audio

    def reset(self) -> None:
        self._buffer.clear()
        self._total_samples = 0


# ── 3. VoiceDetectionEngine Protocol ──────────────────────────────────

class VoiceDetectionEngine(Protocol):
    """Protocol satisfied by any underlying deepfake detector (AASIST, Wav2Vec2, etc.)."""

    def infer_window(self, audio_window: np.ndarray) -> Dict[str, Any]:
        ...

    def classify_score(self, score: float) -> Tuple[ClassificationState, float]:
        ...


# ── 4. DetectionEventHandler & Temporal Smoothing ────────────────────

class DetectionEventHandler:
    """
    Applies temporal smoothing across successive streaming analysis windows,
    prevents single-frame false alarms, tracks call risk trajectories,
    and emits real-time alerts.
    """

    def __init__(
        self,
        call_id: str,
        smoothing_factor: float = 0.35,  # Exponential smoothing alpha
        history_len: int = 10,
        alert_consecutive_windows: int = 2,
        on_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.call_id = call_id
        self.smoothing_factor = smoothing_factor
        self.history_len = history_len
        self.alert_consecutive_windows = alert_consecutive_windows
        self.on_event_callback = on_event_callback

        self.smoothed_score: Optional[float] = None
        self.history: Deque[Dict[str, Any]] = deque(maxlen=history_len)
        self.consecutive_high_risk: int = 0
        self.analysis_count: int = 0

    def process_window_result(
        self,
        raw_score: float,
        classification: ClassificationState,
        latency_ms: float,
        buffered_sec: float
    ) -> Dict[str, Any]:
        """
        Incorporate a new window detection into the smoothed trajectory.
        """
        self.analysis_count += 1
        now = datetime.utcnow()

        # Exponential Moving Average smoothing
        if self.smoothed_score is None:
            self.smoothed_score = raw_score
        else:
            self.smoothed_score = (
                self.smoothing_factor * raw_score + (1.0 - self.smoothing_factor) * self.smoothed_score
            )

        # Evaluate alert trigger: requires sustained elevated score
        is_high = raw_score >= 0.70 or self.smoothed_score >= 0.70
        if is_high:
            self.consecutive_high_risk += 1
        else:
            self.consecutive_high_risk = max(0, self.consecutive_high_risk - 1)

        alert_triggered = self.consecutive_high_risk >= self.alert_consecutive_windows

        event = {
            "call_id": self.call_id,
            "timestamp": now.isoformat(),
            "analysis_index": self.analysis_count,
            "raw_score": round(raw_score, 4),
            "smoothed_score": round(self.smoothed_score, 4),
            "classification": classification.value,
            "alert_triggered": alert_triggered,
            "latency_ms": round(latency_ms, 2),
            "buffered_seconds": round(buffered_sec, 2),
            "consecutive_high_windows": self.consecutive_high_risk
        }

        self.history.append(event)

        if self.on_event_callback:
            try:
                self.on_event_callback(event)
            except Exception as e:
                logger.error(f"Callback error in DetectionEventHandler: {e}")

        return event


# ── 5. CallDetectionSession ───────────────────────────────────────────

class CallDetectionSession:
    """
    End-to-end streaming detection session tying together:
    Source -> Buffer -> Engine -> EventHandler
    """

    def __init__(
        self,
        call_id: str,
        engine: VoiceDetectionEngine,
        sample_rate: int = 16000,
        window_duration_sec: float = 6.0,
        on_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.call_id = call_id
        self.engine = engine
        self.buffer_manager = AudioBufferManager(
            sample_rate=sample_rate,
            window_duration_sec=window_duration_sec,
            overlap_duration_sec=2.0
        )
        self.event_handler = DetectionEventHandler(
            call_id=call_id,
            on_event_callback=on_event_callback
        )
        self.is_active = True

    def feed_pcm(self, pcm_chunk: bytes) -> Optional[Dict[str, Any]]:
        """Feed incoming PCM bytes and run analysis if window is ready."""
        self.buffer_manager.append_pcm_bytes(pcm_chunk)
        if self.buffer_manager.is_window_ready():
            return self._run_analysis_cycle()
        return None

    def feed_array(self, audio_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """Feed incoming numpy array and run analysis if window is ready."""
        self.buffer_manager.append_array(audio_array)
        if self.buffer_manager.is_window_ready():
            return self._run_analysis_cycle()
        return None

    def finalize(self) -> Optional[Dict[str, Any]]:
        """Run analysis on any remaining buffered audio longer than 1 second."""
        remaining = self.buffer_manager.flush_remaining()
        if len(remaining) >= self.buffer_manager.sample_rate:  # at least 1s
            res = self.engine.infer_window(remaining)
            state, _ = self.engine.classify_score(res["deepfake_score"])
            return self.event_handler.process_window_result(
                raw_score=res["deepfake_score"],
                classification=state,
                latency_ms=res["latency_ms"],
                buffered_sec=len(remaining) / self.buffer_manager.sample_rate
            )
        return None

    def _run_analysis_cycle(self) -> Dict[str, Any]:
        """Extract ready window and evaluate with the detection engine."""
        window_audio = self.buffer_manager.extract_window()
        res = self.engine.infer_window(window_audio)
        state, _ = self.engine.classify_score(res["deepfake_score"])

        return self.event_handler.process_window_result(
            raw_score=res["deepfake_score"],
            classification=state,
            latency_ms=res["latency_ms"],
            buffered_sec=self.buffer_manager.buffered_seconds
        )
