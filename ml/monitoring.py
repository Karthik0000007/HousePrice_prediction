"""
Prediction Monitoring
=====================
Production monitoring system for a real-estate analytics deployment.

Tracks:
- Prediction latency (p50, p95, p99)
- Prediction value distribution
- Request volume over time
- Error rates
- Model staleness

Exposes metrics via ``GET /monitoring/metrics`` as structured JSON,
suitable for ingestion by Prometheus/Grafana or internal dashboards.
All state is in-memory with periodic log flushes; for production at
scale, swap with a time-series database (InfluxDB, Prometheus pushgateway).
"""

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PredictionMonitor:
    """Thread-safe monitor that collects per-request telemetry."""

    def __init__(self, max_history: int = 10_000, log_dir: Optional[Path] = None):
        """
        Parameters
        ----------
        max_history : int
            Maximum number of prediction records to keep in memory
            (rolling window).  Older entries are evicted.
        log_dir : Path, optional
            Directory for periodic metric snapshots.
        """
        self._lock = Lock()
        self._max = max_history
        self._log_dir = log_dir

        # Rolling buffers
        self._latencies: deque[float] = deque(maxlen=max_history)
        self._predictions: deque[float] = deque(maxlen=max_history)
        self._timestamps: deque[float] = deque(maxlen=max_history)
        self._errors: deque[float] = deque(maxlen=max_history)

        # Counters (lifetime)
        self.total_requests: int = 0
        self.total_errors: int = 0
        self._start_time: float = time.time()

        logger.info("PredictionMonitor initialised (max_history=%d)", max_history)

    # ── Recording ────────────────────────────────────────────────────────

    def record_prediction(
        self, predicted_price: float, latency_ms: float
    ) -> None:
        """Record a successful prediction."""
        with self._lock:
            now = time.time()
            self._latencies.append(latency_ms)
            self._predictions.append(predicted_price)
            self._timestamps.append(now)
            self.total_requests += 1

    def record_error(self) -> None:
        """Record a failed prediction."""
        with self._lock:
            self._errors.append(time.time())
            self.total_errors += 1
            self.total_requests += 1

    # ── Metrics Snapshot ─────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return a JSON-serializable metrics summary."""
        with self._lock:
            uptime_s = time.time() - self._start_time
            lat = np.array(self._latencies) if self._latencies else np.array([0.0])
            preds = np.array(self._predictions) if self._predictions else np.array([0.0])

            # Requests-per-minute for the last 5 minutes
            now = time.time()
            recent_window = 300  # 5 min
            recent_count = sum(
                1 for t in self._timestamps if now - t <= recent_window
            )
            rpm = recent_count / (recent_window / 60)

            return {
                "uptime_seconds": round(uptime_s, 1),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": round(
                    self.total_errors / max(self.total_requests, 1), 4
                ),
                "requests_per_minute_5m": round(rpm, 2),
                "latency_ms": {
                    "p50": round(float(np.percentile(lat, 50)), 2),
                    "p95": round(float(np.percentile(lat, 95)), 2),
                    "p99": round(float(np.percentile(lat, 99)), 2),
                    "mean": round(float(np.mean(lat)), 2),
                },
                "predictions": {
                    "count_in_buffer": len(self._predictions),
                    "mean": round(float(np.mean(preds)), 2),
                    "min": round(float(np.min(preds)), 2),
                    "max": round(float(np.max(preds)), 2),
                    "std": round(float(np.std(preds)), 2),
                },
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }

    def flush_to_log(self) -> None:
        """Write current metrics to a log file for historical tracking."""
        if self._log_dir is None:
            return
        metrics = self.get_metrics()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        path = self._log_dir / "metrics_history.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        logger.info("Metrics flushed to %s", path)
