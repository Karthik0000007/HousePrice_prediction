"""
Training Pipeline
=================
End-to-end training script that:
1. Loads and audits the Housing dataset
2. Applies feature engineering and preprocessing
3. Splits data into train / validation / test sets
4. Trains a GradientBoostingRegressor with cross-validation
5. Evaluates on held-out test set (MAE, RMSE, R²)
6. Serializes model artifacts for production inference

Usage:
    python -m ml.train          # from project root
    python ml/train.py          # direct execution

Preprocessing Decisions (documented):
    - Binary columns (yes/no) → mapped to 1/0 integers.
    - furnishingstatus → ordinal-encoded (unfurnished=0, semi=1, furnished=2)
      because there is a natural ordering that correlates with price.
    - Engineered features:
        * area_per_bedroom  – captures space efficiency per sleeping area
        * area_per_bathroom – captures bathroom-to-space ratio
        * total_rooms       – bedrooms + bathrooms as overall size proxy
        * luxury_score      – sum of premium binary amenities
          (airconditioning + guestroom + basement + prefarea)
    - StandardScaler applied to all numeric features to normalise ranges.
    - Log-transform of target (price) to reduce skew and stabilise variance;
      predictions are exponentiated back for interpretability.
    - No data leakage: scaler is fit ONLY on training data.
    - No duplicate rows found in raw data (verified at runtime).
    - No missing values in this dataset (verified at runtime).
"""

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Allow running as `python ml/train.py` or `python -m ml.train`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    ARTIFACTS_DIR,
    BINARY_FEATURES,
    BINARY_MAP,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_PATH,
    DRIFT_BASELINE_PATH,
    ENGINEERED_FEATURES,
    FEATURE_NAMES_PATH,
    FEATURE_REGISTRY_PATH,
    FURNISHING_MAP,
    GRADIENT_BOOSTING_PARAMS,
    LOG_DIR,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Data Loading & Audit ────────────────────────────────────────────────
def load_and_audit(path: Path) -> pd.DataFrame:
    """Load CSV and run data-quality checks."""
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Shape: %s rows × %s columns", *df.shape)

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning("Missing values detected:\n%s", missing[missing > 0])
        df = df.dropna()
        logger.info("Rows after dropping missing: %d", len(df))
    else:
        logger.info("No missing values detected [OK]")

    # Duplicates
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        logger.warning("Dropping %d duplicate rows", n_dup)
        df = df.drop_duplicates()
    else:
        logger.info("No duplicate rows [OK]")

    # Price sanity (remove extreme outliers using IQR)
    q1 = df[TARGET].quantile(0.01)
    q99 = df[TARGET].quantile(0.99)
    before = len(df)
    df = df[(df[TARGET] >= q1) & (df[TARGET] <= q99)]
    logger.info("Outlier filter (1st-99th pctl): %d -> %d rows", before, len(df))

    return df.reset_index(drop=True)


# ── Feature Engineering ─────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features and encode categoricals."""
    df = df.copy()

    # Binary encoding
    for col in BINARY_FEATURES:
        df[col] = df[col].map(BINARY_MAP)
        logger.info("Encoded binary column: %s", col)

    # Ordinal encoding for furnishing status
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].map(FURNISHING_MAP)
        logger.info("Ordinal-encoded: %s", col)

    # Engineered features
    df["area_per_bedroom"] = df["area"] / (df["bedrooms"] + 1)
    df["area_per_bathroom"] = df["area"] / (df["bathrooms"] + 1)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["luxury_score"] = (
        df["airconditioning"] + df["guestroom"] + df["basement"] + df["prefarea"]
    )

    logger.info("Engineered %d new features", len(ENGINEERED_FEATURES))
    return df


# ── Build Feature Matrix ────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame):
    """Return X, y (log-transformed target), and feature names."""
    feature_cols = (
        NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
    )
    X = df[feature_cols].values
    y = np.log1p(df[TARGET].values)  # log-transform target
    return X, y, feature_cols


# ── Main Training Pipeline ──────────────────────────────────────────────
def train():
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    # 1. Load & audit
    df = load_and_audit(DATA_PATH)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Build matrices
    X, y, feature_names = build_feature_matrix(df)
    logger.info("Feature matrix shape: %s", X.shape)

    # 4. Train / Temp split  (temp = val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE, random_state=RANDOM_STATE
    )
    # Split temp into val and test
    relative_test = TEST_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=RANDOM_STATE
    )
    logger.info(
        "Split sizes -- Train: %d | Val: %d | Test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # 5. Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 6. Cross-validation on training set
    model = GradientBoostingRegressor(**GRADIENT_BOOSTING_PARAMS)
    cv_scores = cross_val_score(
        model, X_train_s, y_train, cv=CV_FOLDS, scoring="r2"
    )
    logger.info(
        "Cross-Validation R2 (mean +/- std): %.4f +/- %.4f",
        cv_scores.mean(), cv_scores.std(),
    )

    # 7. Train final model
    model.fit(X_train_s, y_train)

    # 8. Validation metrics
    y_val_pred = model.predict(X_val_s)
    val_metrics = _compute_metrics(y_val, y_val_pred, "Validation")

    # 9. Test metrics
    y_test_pred = model.predict(X_test_s)
    test_metrics = _compute_metrics(y_test, y_test_pred, "Test")

    # 10. Feature importance
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info("Top-10 Feature Importances:")
    for name, imp in importances[:10]:
        logger.info("  %-25s %.4f", name, imp)

    # 11. Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    logger.info("Model saved  -> %s", MODEL_PATH)
    logger.info("Scaler saved -> %s", SCALER_PATH)

    # 12. Save drift detection baseline (training set, unscaled)
    from ml.drift import save_baseline
    save_baseline(X_train, DRIFT_BASELINE_PATH)

    # 13. Register feature version
    from ml.feature_registry import (
        FeatureRegistry, build_current_version,
    )
    registry = FeatureRegistry(FEATURE_REGISTRY_PATH)
    fv = build_current_version()
    try:
        registry.register_version(fv)
    except ValueError:
        # Version already registered (re-run); update model compatibility
        logger.info("Feature version %s already registered", fv.version)

    # Compute residuals for confidence interval estimation
    residuals = y_test - y_test_pred
    residual_std = float(np.std(residuals))

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "residual_std": residual_std,
        "model_type": "GradientBoostingRegressor",
        "hyperparameters": GRADIENT_BOOSTING_PARAMS,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metadata saved -> %s", METADATA_PATH)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)


def _compute_metrics(y_true, y_pred, split_name: str) -> dict:
    """Compute and log regression metrics on log-scale and original scale."""
    # Metrics in log-space
    mae_log = mean_absolute_error(y_true, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Metrics in original price space
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))

    logger.info("-- %s Metrics --", split_name)
    logger.info("  R²   : %.4f", r2)
    logger.info("  MAE  : %.2f (original scale)", mae_orig)
    logger.info("  RMSE : %.2f (original scale)", rmse_orig)
    logger.info("  MAE  : %.4f (log scale)", mae_log)
    logger.info("  RMSE : %.4f (log scale)", rmse_log)

    return {
        "r2": float(r2),
        "mae": float(mae_orig),
        "rmse": float(rmse_orig),
        "mae_log": float(mae_log),
        "rmse_log": float(rmse_log),
    }


if __name__ == "__main__":
    train()
