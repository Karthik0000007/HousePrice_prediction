"""
Preprocessing Module for Inference
====================================
Mirror of the training-time preprocessing, applied to a single
property input dictionary for real-time inference.
"""

import numpy as np
from ml.config import (
    BINARY_FEATURES,
    BINARY_MAP,
    CATEGORICAL_FEATURES,
    FURNISHING_MAP,
    NUMERIC_FEATURES,
    ENGINEERED_FEATURES,
)


def preprocess_input(data: dict) -> np.ndarray:
    """
    Convert a validated property input dict into a feature vector
    matching the training feature order.

    Parameters
    ----------
    data : dict
        Raw input with keys matching PropertyInput fields.

    Returns
    -------
    np.ndarray
        1-D array of shape (n_features,) ready for scaling.
    """
    row = {}

    # Numeric features (pass-through)
    for col in NUMERIC_FEATURES:
        row[col] = float(data[col])

    # Binary features
    for col in BINARY_FEATURES:
        row[col] = float(BINARY_MAP[data[col]])

    # Categorical features
    for col in CATEGORICAL_FEATURES:
        row[col] = float(FURNISHING_MAP[data[col]])

    # Engineered features
    row["area_per_bedroom"] = row["area"] / (row["bedrooms"] + 1)
    row["area_per_bathroom"] = row["area"] / (row["bathrooms"] + 1)
    row["total_rooms"] = row["bedrooms"] + row["bathrooms"]
    row["luxury_score"] = (
        row["airconditioning"]
        + row["guestroom"]
        + row["basement"]
        + row["prefarea"]
    )

    # Assemble in exact training order
    feature_order = (
        NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
    )
    vector = np.array([row[f] for f in feature_order], dtype=np.float64)
    return vector
