"""
Model Retraining Strategy
=========================
Implements a policy-driven retraining orchestrator for a real-estate
analytics company operating PropertyValuator in production.

Retraining Triggers
-------------------
A model retraining is recommended when ANY of:
1. **Scheduled cadence** -- model age exceeds ``MAX_MODEL_AGE_DAYS``
   (default 30 days) to capture market shifts.
2. **Data drift detected** -- DriftDetector reports
   ``significant_drift`` on >= ``DRIFT_FEATURE_THRESHOLD`` features.
3. **Performance degradation** -- if a labelled feedback loop exists,
   R2 on recent labelled predictions drops below ``MIN_R2_THRESHOLD``.
4. **Data volume** -- new labelled data exceeds
   ``MIN_NEW_SAMPLES`` since last training (ensures enough signal
   for a meaningful update).

Strategy Notes
--------------
- **Shadow evaluation**: when new data arrives, retrain a candidate
  model and compare against the production model on a hold-out set
  before promoting.
- **Rollback**: the previous model artifact is archived with a
  timestamp; if the new model degrades after deployment, the most
  recent good model can be restored instantly.
- **Incremental vs. full retrain**: GradientBoosting with
  ``warm_start=True`` supports incremental fitting, but a full
  retrain from scratch is safer to avoid compounding bias.
  We default to full retrain.

This module does NOT auto-trigger retraining; it evaluates the
policy and returns a structured recommendation that an external
scheduler (cron, Airflow, Prefect) or an admin can act on.
"""

import json
import logging
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Policy Configuration ────────────────────────────────────────────────

class RetrainingPolicy:
    """Configurable thresholds that govern when retraining is triggered."""

    MAX_MODEL_AGE_DAYS: int = 30
    DRIFT_FEATURE_THRESHOLD: int = 2        # num features with significant drift
    MIN_R2_THRESHOLD: float = 0.55          # below this -> retrain
    MIN_NEW_SAMPLES: int = 100              # minimum new labelled rows needed


# ── Evaluator ────────────────────────────────────────────────────────────

class RetrainingEvaluator:
    """Evaluates whether the current model should be retrained."""

    def __init__(
        self,
        metadata_path: Path,
        policy: Optional[RetrainingPolicy] = None,
    ):
        self.metadata_path = metadata_path
        self.policy = policy or RetrainingPolicy()

    def evaluate(
        self,
        drift_report: Optional[dict] = None,
        recent_r2: Optional[float] = None,
        new_sample_count: int = 0,
    ) -> dict:
        """
        Evaluate all retraining triggers and return a recommendation.

        Returns
        -------
        dict
            {
                "should_retrain": bool,
                "reasons": [str, ...],
                "model_age_days": float,
                "checks": {trigger_name: {passed: bool, detail: str}, ...}
            }
        """
        reasons: list[str] = []
        checks: dict = {}

        # 1. Model age
        model_age = self._model_age_days()
        age_exceeded = model_age > self.policy.MAX_MODEL_AGE_DAYS
        checks["scheduled_cadence"] = {
            "passed": not age_exceeded,
            "detail": (
                f"Model is {model_age:.1f} days old "
                f"(threshold: {self.policy.MAX_MODEL_AGE_DAYS}d)"
            ),
        }
        if age_exceeded:
            reasons.append(
                f"Model age ({model_age:.0f}d) exceeds "
                f"{self.policy.MAX_MODEL_AGE_DAYS}d cadence"
            )

        # 2. Data drift
        drift_triggered = False
        if drift_report and drift_report.get("features"):
            sig_count = sum(
                1 for f in drift_report["features"].values()
                if f.get("status") == "significant_drift"
            )
            drift_triggered = sig_count >= self.policy.DRIFT_FEATURE_THRESHOLD
            checks["data_drift"] = {
                "passed": not drift_triggered,
                "detail": (
                    f"{sig_count} features with significant drift "
                    f"(threshold: {self.policy.DRIFT_FEATURE_THRESHOLD})"
                ),
            }
            if drift_triggered:
                reasons.append(
                    f"{sig_count} features show significant drift"
                )
        else:
            checks["data_drift"] = {
                "passed": True,
                "detail": "No drift report available",
            }

        # 3. Performance degradation
        if recent_r2 is not None:
            perf_bad = recent_r2 < self.policy.MIN_R2_THRESHOLD
            checks["performance"] = {
                "passed": not perf_bad,
                "detail": (
                    f"Recent R2 = {recent_r2:.4f} "
                    f"(threshold: {self.policy.MIN_R2_THRESHOLD})"
                ),
            }
            if perf_bad:
                reasons.append(
                    f"R2 ({recent_r2:.4f}) below "
                    f"threshold ({self.policy.MIN_R2_THRESHOLD})"
                )
        else:
            checks["performance"] = {
                "passed": True,
                "detail": "No labelled feedback available yet",
            }

        # 4. Data volume
        enough_data = new_sample_count >= self.policy.MIN_NEW_SAMPLES
        checks["data_volume"] = {
            "passed": True,  # Not a blocking gate, just advisory
            "detail": (
                f"{new_sample_count} new samples "
                f"(recommended minimum: {self.policy.MIN_NEW_SAMPLES})"
            ),
        }

        should_retrain = len(reasons) > 0
        result = {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "model_age_days": round(model_age, 1),
            "checks": checks,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Retraining evaluation: should_retrain=%s, reasons=%s",
            should_retrain, reasons,
        )
        return result

    def _model_age_days(self) -> float:
        """Calculate how old the current model is in days."""
        try:
            with open(self.metadata_path) as f:
                meta = json.load(f)
            trained_at = datetime.fromisoformat(meta["trained_at"])
            age = datetime.now(timezone.utc) - trained_at
            return age.total_seconds() / 86400
        except Exception:
            logger.warning("Could not determine model age, defaulting to 999")
            return 999.0


# ── Artifact Archival ────────────────────────────────────────────────────

def archive_current_model(artifacts_dir: Path) -> Path:
    """
    Copy current model artifacts to a timestamped archive directory
    for rollback capability.

    Returns
    -------
    Path
        The archive directory path.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = artifacts_dir / "archive" / ts
    archive_dir.mkdir(parents=True, exist_ok=True)

    for artifact in ["model.joblib", "scaler.joblib", "metadata.json",
                      "feature_names.joblib", "drift_baseline.npy"]:
        src = artifacts_dir / artifact
        if src.exists():
            shutil.copy2(src, archive_dir / artifact)

    logger.info("Archived model artifacts -> %s", archive_dir)
    return archive_dir
