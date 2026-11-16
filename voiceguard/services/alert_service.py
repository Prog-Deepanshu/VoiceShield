"""
VoiceGuard — Alert Service
============================
Dispatches fraud alerts to configured banking system endpoints
when high-risk deepfake activity is detected.
"""

import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class AlertService:
    """
    Fraud alert dispatch service.

    Sends webhook notifications to the banking system's fraud
    prevention infrastructure when a high-risk deepfake is detected.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        timeout_sec: int = 10,
    ):
        """
        Args:
            webhook_url:  URL to POST alert notifications to
            timeout_sec:  HTTP request timeout in seconds
        """
        self.webhook_url = webhook_url
        self.timeout_sec = timeout_sec
        self._alert_count = 0

    async def send_alert_async(
        self,
        call_id: str,
        score: float,
        risk_level: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Send an async fraud alert via webhook.

        Args:
            call_id:    Unique call identifier
            score:      Deepfake probability score
            risk_level: Risk classification (LOW/MEDIUM/HIGH)
            metadata:   Additional alert metadata

        Returns:
            True if alert was sent successfully
        """
        if not self.webhook_url:
            logger.debug(
                f"Alert skipped (no webhook configured): "
                f"call_id={call_id}, score={score:.3f}"
            )
            return False

        payload = {
            "alert_type": "VOICE_DEEPFAKE_DETECTED",
            "call_id": call_id,
            "deepfake_score": round(score, 4),
            "risk_level": risk_level,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "VoiceGuard/1.0.0",
            "metadata": metadata or {},
        }

        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Alert-Source": "VoiceGuard",
                    },
                )

                if response.status_code < 300:
                    self._alert_count += 1
                    logger.info(
                        f"Alert sent successfully: call_id={call_id}, "
                        f"score={score:.3f}, status={response.status_code}"
                    )
                    return True
                else:
                    logger.warning(
                        f"Alert webhook returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    return False

        except ImportError:
            logger.error(
                "httpx library required for webhook alerts. "
                "Install with: pip install httpx"
            )
            return False
        except Exception as e:
            logger.error(f"Alert dispatch failed: {e}")
            return False

    def send_alert_sync(
        self,
        call_id: str,
        score: float,
        risk_level: str,
    ) -> bool:
        """
        Synchronous alert dispatch (for non-async contexts).
        """
        if not self.webhook_url:
            return False

        payload = {
            "alert_type": "VOICE_DEEPFAKE_DETECTED",
            "call_id": call_id,
            "deepfake_score": round(score, 4),
            "risk_level": risk_level,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "VoiceGuard/1.0.0",
        }

        try:
            import httpx

            with httpx.Client(timeout=self.timeout_sec) as client:
                response = client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                return response.status_code < 300

        except Exception as e:
            logger.error(f"Sync alert dispatch failed: {e}")
            return False

    @property
    def alert_count(self) -> int:
        return self._alert_count
