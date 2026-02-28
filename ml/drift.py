"""
Data Drift Detection
====================
Detects distributional shifts between training data and incoming
prediction requests using two statistical methods:

1. **Population Stability Index (PSI)** -- measures shift in the
   distribution of each numeric feature by binning values into
   quantile-based buckets and comparing observed vs. expected
   proportions.  PSI < 0.1 = stable, 0.1-0.25 = moderate drift,
   > 0.25 = significant drift.

2. **Kolmogorov-Smirnov (KS) Test** -- non-parametric test that
   compares the empirical CDFs of the training baseline and the
   incoming sample.  A p-value < 0.05 indicates statistically
   significant drift.

Usage:
    The DriftDetector is initialised once at server startup with
    the training-set reference distributions.  Every N predictions
    (configurable), the buffered inputs are evaluated for drift.
    Results are exposed via ``GET /monitoring/drift``.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Stateful detector that buffers incoming feature vectors and
    periodically compares them against training-time baselines."""

    # PSI thresholds
    PSI_STABLE = 0.10
    PSI_MODERATE = 0.25

    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: list[str],
        n_bins: int = 10,
        buffer_size: int = 50,
    ):
        """
        Parameters
        ----------
        reference_data : np.ndarray
            Training feature matrix (n_samples, n_features).
        feature_names : list[str]
            Ordered feature names matching columns of reference_data.
        n_bins : int
            Number of quantile bins for PSI calculation.
        buffer_size : int
            Minimum number of buffered predictions before drift
            analysis is triggered.
        """
        self.feature_names = feature_names
        self.n_bins = n_bins
        self.buffer_size = buffer_size

        # Compute reference quantile edges per feature
        self._ref_data = reference_data
        self._ref_edges: list[np.ndarray] = []
        self._ref_proportions: list[np.ndarray] = []
        for col_idx in range(reference_data.shape[1]):
            col = reference_data[:, col_idx]
            edges = np.quantile(col, np.linspace(0, 1, n_bins + 1))
            edges = np.unique(edges)  # collapse duplicate edges
            counts = np.histogram(col, bins=edges)[0]
            props = counts / counts.sum()
            props = np.clip(props, 1e-6, None)  # avoid log(0)
            self._ref_edges.append(edges)
            self._ref_proportions.append(props)

        # Prediction buffer
        self._buffer: list[np.ndarray] = []
        self._last_report: Optional[dict] = None
        logger.info(
            "DriftDetector initialised with %d reference samples, "
            "%d features, buffer_size=%d",
            len(reference_data), len(feature_names), buffer_size,
        )

    # ── Public Interface ─────────────────────────────────────────────────

    def record(self, feature_vector: np.ndarray) -> None:
        """Buffer a single prediction's feature vector."""
        self._buffer.append(feature_vector.copy())

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    def analyze(self, force: bool = False) -> Optional[dict]:
        """Run drift analysis if enough samples are buffered.

        Parameters
        ----------
        force : bool
            Run even if buffer_size threshold not met.

        Returns
        -------
        dict or None
            Per-feature PSI and KS results, plus overall status.
        """
        if not force and len(self._buffer) < self.buffer_size:
            return self._last_report

        if len(self._buffer) == 0:
            return {"status": "no_data", "features": {}}

        sample = np.vstack(self._buffer)
        report = {"n_samples": len(sample), "features": {}}

        overall_status = "stable"
        for col_idx, fname in enumerate(self.feature_names):
            col = sample[:, col_idx]
            psi = self._psi(col_idx, col)
            ks_stat, ks_pvalue = stats.ks_2samp(
                self._ref_data[:, col_idx], col
            )

            if psi > self.PSI_MODERATE:
                feat_status = "significant_drift"
                overall_status = "significant_drift"
            elif psi > self.PSI_STABLE:
                feat_status = "moderate_drift"
                if overall_status == "stable":
                    overall_status = "moderate_drift"
            else:
                feat_status = "stable"

            report["features"][fname] = {
                "psi": round(float(psi), 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 4),
                "status": feat_status,
            }

        report["status"] = overall_status
        self._last_report = report
        logger.info(
            "Drift analysis complete: status=%s, n_samples=%d",
            overall_status, len(sample),
        )
        return report

    def flush_buffer(self) -> None:
        """Clear the prediction buffer after analysis."""
        self._buffer.clear()

    def get_last_report(self) -> Optional[dict]:
        return self._last_report

    # ── Internal ─────────────────────────────────────────────────────────

    def _psi(self, col_idx: int, observed: np.ndarray) -> float:
        """Compute Population Stability Index for one feature."""
        edges = self._ref_edges[col_idx]
        expected = self._ref_proportions[col_idx]

        counts = np.histogram(observed, bins=edges)[0]
        actual = counts / max(counts.sum(), 1)
        actual = np.clip(actual, 1e-6, None)

        # Align lengths when edge collapsing changed bin count
        min_len = min(len(expected), len(actual))
        expected = expected[:min_len]
        actual = actual[:min_len]

        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        return psi


# ── Baseline Persistence ────────────────────────────────────────────────

def save_baseline(data: np.ndarray, path: Path) -> None:
    """Persist training reference data for drift detection at startup."""
    np.save(str(path), data)
    logger.info("Drift baseline saved -> %s", path)


def load_baseline(path: Path) -> np.ndarray:
    """Load training reference data from disk."""
    data = np.load(str(path))
    logger.info("Drift baseline loaded <- %s (%d samples)", path, len(data))
    return data
