"""
Unit tests for the Real-Time Call Protection architecture:
AudioBufferManager, DetectionEventHandler, and temporal smoothing.
"""

import numpy as np
import pytest

from app.models.schemas import ClassificationState
from app.services.call_protection import (
    AudioBufferManager,
    DetectionEventHandler,
)


def test_audio_buffer_manager():
    sr = 16000
    mgr = AudioBufferManager(
        sample_rate=sr,
        window_duration_sec=4.0,
        overlap_duration_sec=1.0,
        dtype="float32"
    )

    # Initially empty
    assert not mgr.is_window_ready()
    assert mgr.buffered_seconds == 0.0

    # Feed 2 seconds
    chunk_2s = np.random.randn(sr * 2).astype(np.float32)
    mgr.append_array(chunk_2s)
    assert not mgr.is_window_ready()
    assert mgr.buffered_seconds == pytest.approx(2.0, 0.01)

    # Feed another 3 seconds (total 5s >= 4s window)
    chunk_3s = np.random.randn(sr * 3).astype(np.float32)
    mgr.append_array(chunk_3s)
    assert mgr.is_window_ready()
    assert mgr.buffered_seconds == pytest.approx(5.0, 0.01)

    # Extract window
    window = mgr.extract_window()
    assert len(window) == sr * 4
    # Overlap retention check: buffer should now retain 2.0s (5.0s - 4.0s + 1.0s overlap)
    assert mgr.buffered_seconds == pytest.approx(2.0, 0.01)


def test_detection_event_handler_smoothing():
    events_dispatched = []

    handler = DetectionEventHandler(
        call_id="TEST-001",
        smoothing_factor=0.5,
        alert_consecutive_windows=2,
        on_event_callback=lambda evt: events_dispatched.append(evt)
    )

    # Window 1: High raw score (0.80)
    evt1 = handler.process_window_result(
        raw_score=0.80,
        classification=ClassificationState.LIKELY_AI_GENERATED,
        latency_ms=45.0,
        buffered_sec=4.0
    )
    assert evt1["smoothed_score"] == 0.80
    assert not evt1["alert_triggered"]  # needs 2 consecutive windows

    # Window 2: Sustained high raw score (0.84)
    evt2 = handler.process_window_result(
        raw_score=0.84,
        classification=ClassificationState.LIKELY_AI_GENERATED,
        latency_ms=42.0,
        buffered_sec=4.0
    )
    assert evt2["smoothed_score"] == pytest.approx(0.82, 0.01)
    assert evt2["alert_triggered"]  # now 2 consecutive windows -> alert fires!

    # Window 3: Low raw score (0.10)
    evt3 = handler.process_window_result(
        raw_score=0.10,
        classification=ClassificationState.LIKELY_HUMAN,
        latency_ms=40.0,
        buffered_sec=4.0
    )
    assert evt3["smoothed_score"] == pytest.approx(0.46, 0.01)
    assert not evt3["alert_triggered"]

    assert len(events_dispatched) == 3
