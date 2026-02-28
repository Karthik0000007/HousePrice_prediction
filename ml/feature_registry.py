"""
Feature Registry & Versioning
==============================
Provides a lightweight feature versioning system so a real-estate
analytics company can:

1. **Track schema evolution** -- every change to the feature set
   (additions, removals, encoding changes) is recorded with a
   semantic version and a human-readable changelog.

2. **Guarantee inference/training parity** -- the registry is the
   single source of truth for what features exist, their types,
   and their transformations.  Both ``ml/train.py`` and
   ``app/preprocessing.py`` derive their logic from ``ml/config.py``,
   which references the active version in this registry.

3. **Support rollback** -- if a new feature version degrades model
   quality, revert to a prior version by re-pointing the active
   version in the registry.

Schema
------
Each version entry contains:
- ``version``     : semver string (MAJOR.MINOR.PATCH)
- ``created_at``  : ISO-8601 timestamp
- ``features``    : ordered list of feature descriptors
- ``changelog``   : human-readable description of changes
- ``compatible_model_versions`` : list of model artifact timestamps
  that were trained with this feature set

The registry is persisted as a single JSON file in ``artifacts/``.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Feature Descriptor ──────────────────────────────────────────────────

class FeatureDescriptor:
    """Describes a single feature in the schema."""

    def __init__(
        self,
        name: str,
        dtype: str,
        source: str,
        transform: str,
        description: str = "",
    ):
        self.name = name
        self.dtype = dtype
        self.source = source          # "raw" | "engineered"
        self.transform = transform    # "none" | "standard_scaler" | "binary_map" | etc
        self.description = description

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "source": self.source,
            "transform": self.transform,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureDescriptor":
        return cls(**d)


# ── Feature Version ─────────────────────────────────────────────────────

class FeatureVersion:
    """An immutable snapshot of the feature schema at a point in time."""

    def __init__(
        self,
        version: str,
        features: list[FeatureDescriptor],
        changelog: str,
        created_at: Optional[str] = None,
        compatible_model_versions: Optional[list[str]] = None,
    ):
        self.version = version
        self.features = features
        self.changelog = changelog
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.compatible_model_versions = compatible_model_versions or []

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "changelog": self.changelog,
            "compatible_model_versions": self.compatible_model_versions,
            "features": [f.to_dict() for f in self.features],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureVersion":
        features = [FeatureDescriptor.from_dict(f) for f in d["features"]]
        return cls(
            version=d["version"],
            features=features,
            changelog=d["changelog"],
            created_at=d.get("created_at"),
            compatible_model_versions=d.get("compatible_model_versions", []),
        )


# ── Feature Registry ────────────────────────────────────────────────────

class FeatureRegistry:
    """Manages the full history of feature schema versions."""

    def __init__(self, registry_path: Path):
        self.path = registry_path
        self._versions: list[FeatureVersion] = []
        self._active_version: Optional[str] = None

        if self.path.exists():
            self._load()
        else:
            logger.info("Feature registry not found; will create on first save")

    # ── CRUD ─────────────────────────────────────────────────────────────

    def register_version(self, version: FeatureVersion) -> None:
        """Add a new feature version to the registry."""
        existing = [v.version for v in self._versions]
        if version.version in existing:
            raise ValueError(f"Version {version.version} already exists")
        self._versions.append(version)
        self._active_version = version.version
        self._save()
        logger.info(
            "Registered feature version %s (%d features)",
            version.version, len(version.features),
        )

    def get_active(self) -> Optional[FeatureVersion]:
        """Return the currently active feature version."""
        if not self._active_version:
            return None
        return self.get_version(self._active_version)

    def get_version(self, version: str) -> Optional[FeatureVersion]:
        for v in self._versions:
            if v.version == version:
                return v
        return None

    def list_versions(self) -> list[dict]:
        """Return a summary of all registered versions."""
        return [
            {
                "version": v.version,
                "created_at": v.created_at,
                "n_features": len(v.features),
                "changelog": v.changelog,
                "is_active": v.version == self._active_version,
            }
            for v in self._versions
        ]

    def set_active(self, version: str) -> None:
        """Switch the active feature version (e.g., for rollback)."""
        if not any(v.version == version for v in self._versions):
            raise ValueError(f"Version {version} not found in registry")
        self._active_version = version
        self._save()
        logger.info("Active feature version set to %s", version)

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_version": self._active_version,
            "versions": [v.to_dict() for v in self._versions],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        self._active_version = data.get("active_version")
        self._versions = [
            FeatureVersion.from_dict(v) for v in data.get("versions", [])
        ]
        logger.info(
            "Feature registry loaded: %d versions, active=%s",
            len(self._versions), self._active_version,
        )


# ── Convenience: build current version from config ──────────────────────

def build_current_version() -> FeatureVersion:
    """Construct a FeatureVersion from the current ml.config definitions."""
    from ml.config import (
        NUMERIC_FEATURES,
        BINARY_FEATURES,
        CATEGORICAL_FEATURES,
        ENGINEERED_FEATURES,
    )

    features = []

    for name in NUMERIC_FEATURES:
        features.append(FeatureDescriptor(
            name=name, dtype="float64", source="raw",
            transform="standard_scaler",
            description=f"Raw numeric feature: {name}",
        ))

    for name in BINARY_FEATURES:
        features.append(FeatureDescriptor(
            name=name, dtype="int", source="raw",
            transform="binary_map(yes=1,no=0)",
            description=f"Binary indicator: {name}",
        ))

    for name in CATEGORICAL_FEATURES:
        features.append(FeatureDescriptor(
            name=name, dtype="int", source="raw",
            transform="ordinal(unfurnished=0,semi=1,furnished=2)",
            description=f"Ordinal categorical: {name}",
        ))

    eng_descriptions = {
        "area_per_bedroom": "area / (bedrooms + 1)",
        "area_per_bathroom": "area / (bathrooms + 1)",
        "total_rooms": "bedrooms + bathrooms",
        "luxury_score": "AC + guestroom + basement + prefarea",
    }
    for name in ENGINEERED_FEATURES:
        features.append(FeatureDescriptor(
            name=name, dtype="float64", source="engineered",
            transform="derived",
            description=eng_descriptions.get(name, ""),
        ))

    return FeatureVersion(
        version="1.0.0",
        features=features,
        changelog="Initial feature set: 5 numeric, 6 binary, 1 ordinal, 4 engineered",
    )
