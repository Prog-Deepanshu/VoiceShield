"""
VoiceGuard — Audit Logger
===========================
Compliance-grade immutable audit logging for all deepfake analyses.

Logs are written as JSON Lines (.jsonl) for easy ingestion into
SIEM systems, data pipelines, and forensic analysis tools.

Privacy: No raw audio is stored — only metadata and scores.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Immutable audit logger for deepfake analysis events.

    Each log entry is a JSON line containing:
    - Timestamp (ISO 8601)
    - Call ID
    - Deepfake score
    - Risk level and action
    - Models used and latency
    - No raw audio (privacy compliance)
    """

    def __init__(self, log_path: str = "./logs/audit.jsonl"):
        """
        Args:
            log_path: Path to the audit log file
        """
        self.log_path = Path(log_path)
        self._lock = Lock()
        self._file = None
        self._entry_count = 0

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Open log file in append mode
        self._file = open(self.log_path, "a", encoding="utf-8")
        logger.info(f"Audit log initialized: {self.log_path}")

    def log_analysis(
        self,
        call_id: str,
        score: float,
        risk_level: str,
        action: str,
        latency_ms: float,
        models_used: list[str],
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Log a deepfake analysis event.

        Args:
            call_id:     Unique call identifier
            score:       Deepfake probability score [0, 1]
            risk_level:  Risk classification (LOW/MEDIUM/HIGH)
            action:      Recommended action (ALLOW/FLAG_FOR_REVIEW/BLOCK_AND_ALERT)
            latency_ms:  Inference latency in milliseconds
            models_used: List of model names used
            metadata:    Optional additional metadata
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "deepfake_analysis",
            "call_id": call_id,
            "deepfake_score": round(score, 4),
            "risk_level": risk_level,
            "recommended_action": action,
            "inference_latency_ms": round(latency_ms, 2),
            "models_used": models_used,
            "service_version": "1.0.0",
        }

        if metadata:
            entry["metadata"] = metadata

        self._write_entry(entry)

    def log_alert(
        self,
        call_id: str,
        alert_type: str,
        success: bool,
        details: str = "",
    ) -> None:
        """Log an alert dispatch event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "alert_dispatch",
            "call_id": call_id,
            "alert_type": alert_type,
            "success": success,
            "details": details,
        }
        self._write_entry(entry)

    def log_system_event(self, event: str, details: str = "") -> None:
        """Log a system-level event (startup, shutdown, errors)."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "system",
            "event": event,
            "details": details,
        }
        self._write_entry(entry)

    def _write_entry(self, entry: dict) -> None:
        """Write a single JSON line to the audit log (thread-safe)."""
        with self._lock:
            try:
                line = json.dumps(entry, ensure_ascii=False)
                self._file.write(line + "\n")
                self._file.flush()
                self._entry_count += 1
            except Exception as e:
                logger.error(f"Failed to write audit log entry: {e}")

    @property
    def entry_count(self) -> int:
        return self._entry_count

    def close(self) -> None:
        """Close the audit log file."""
        with self._lock:
            if self._file and not self._file.closed:
                self.log_system_event(
                    "shutdown",
                    f"Total entries logged: {self._entry_count}",
                )
                self._file.close()
                logger.info(
                    f"Audit log closed. Total entries: {self._entry_count}"
                )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
