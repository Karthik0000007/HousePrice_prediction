"""
ML Pipeline Configuration
=========================
Central configuration for feature definitions, model hyperparameters,
and preprocessing constants used across training and inference.
"""

from pathlib import Path

# -- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Housing.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# -- Feature Definitions ---------------------------------------------------
TARGET = "price"

NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]

BINARY_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]

CATEGORICAL_FEATURES = ["furnishingstatus"]

# Mapping for binary yes/no columns
BINARY_MAP = {"yes": 1, "no": 0}

# Ordinal mapping for furnishing status (unfurnished < semi < furnished)
FURNISHING_MAP = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}

# -- Engineered Features ---------------------------------------------------
ENGINEERED_FEATURES = [
    "area_per_bedroom",
    "area_per_bathroom",
    "total_rooms",
    "luxury_score",
]

# -- Model Hyperparameters -------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of the remaining training set
CV_FOLDS = 5

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
}
