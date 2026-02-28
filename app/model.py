"""
Model Loader & Inference Service
==================================
Loads serialized artifacts once at startup and exposes a
`predict()` function for real-time property valuation.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

from ml.config import ARTIFACTS_DIR, METADATA_PATH, MODEL_PATH, SCALER_PATH

logger = logging.getLogger(__name__)

# ── Module-level singletons (loaded once) ────────────────────────────────
_model = None
_scaler = None
_metadata = None


def load_artifacts() -> None:
    """Load model, scaler, and metadata from disk into memory."""
    global _model, _scaler, _metadata

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python -m ml.train` first."
        )

    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)

    with open(METADATA_PATH) as f:
        _metadata = json.load(f)

    logger.info("Model artifacts loaded successfully (trained %s)", _metadata.get("trained_at"))


def get_metadata() -> dict:
    if _metadata is None:
        load_artifacts()
    return _metadata


def predict(feature_vector: np.ndarray) -> dict:
    """
    Run inference on a single feature vector.

    Parameters
    ----------
    feature_vector : np.ndarray
        Shape (n_features,) — raw (unscaled) feature values.

    Returns
    -------
    dict with predicted_price, confidence_low, confidence_high,
    currency, model_version.
    """
    if _model is None:
        load_artifacts()

    # Scale
    X = _scaler.transform(feature_vector.reshape(1, -1))

    # Predict (log-space)
    y_log = _model.predict(X)[0]

    # Confidence interval using residual std from training
    residual_std = _metadata.get("residual_std", 0.15)
    z = 1.645  # 90% CI
    low_log = y_log - z * residual_std
    high_log = y_log + z * residual_std

    # Transform back to original price scale
    predicted_price = float(np.expm1(y_log))
    confidence_low = float(np.expm1(low_log))
    confidence_high = float(np.expm1(high_log))

    return {
        "predicted_price": round(predicted_price, 2),
        "confidence_low": round(confidence_low, 2),
        "confidence_high": round(confidence_high, 2),
        "currency": "INR",
        "model_version": _metadata.get("trained_at", "unknown"),
    }
