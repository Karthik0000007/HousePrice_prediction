"""
Pydantic Schemas for Request / Response Validation
===================================================
Defines strict input validation with sensible bounds for all
property features, plus structured prediction responses.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional


class PropertyInput(BaseModel):
    """Input schema for a property valuation request."""

    area: float = Field(
        ..., gt=100, le=50000,
        description="Total area in square feet (100–50,000)",
        json_schema_extra={"example": 5000},
    )
    bedrooms: int = Field(
        ..., ge=1, le=10,
        description="Number of bedrooms (1–10)",
        json_schema_extra={"example": 3},
    )
    bathrooms: int = Field(
        ..., ge=1, le=8,
        description="Number of bathrooms (1–8)",
        json_schema_extra={"example": 2},
    )
    stories: int = Field(
        ..., ge=1, le=5,
        description="Number of stories (1–5)",
        json_schema_extra={"example": 2},
    )
    mainroad: str = Field(
        ..., pattern="^(yes|no)$",
        description="Connected to main road (yes/no)",
        json_schema_extra={"example": "yes"},
    )
    guestroom: str = Field(
        ..., pattern="^(yes|no)$",
        description="Has guest room (yes/no)",
        json_schema_extra={"example": "no"},
    )
    basement: str = Field(
        ..., pattern="^(yes|no)$",
        description="Has basement (yes/no)",
        json_schema_extra={"example": "no"},
    )
    hotwaterheating: str = Field(
        ..., pattern="^(yes|no)$",
        description="Has hot water heating (yes/no)",
        json_schema_extra={"example": "no"},
    )
    airconditioning: str = Field(
        ..., pattern="^(yes|no)$",
        description="Has air conditioning (yes/no)",
        json_schema_extra={"example": "yes"},
    )
    parking: int = Field(
        ..., ge=0, le=5,
        description="Number of parking spaces (0–5)",
        json_schema_extra={"example": 2},
    )
    prefarea: str = Field(
        ..., pattern="^(yes|no)$",
        description="In preferred area (yes/no)",
        json_schema_extra={"example": "yes"},
    )
    furnishingstatus: str = Field(
        ..., pattern="^(furnished|semi-furnished|unfurnished)$",
        description="Furnishing status",
        json_schema_extra={"example": "semi-furnished"},
    )


class PredictionResponse(BaseModel):
    """Structured prediction output with confidence range."""

    predicted_price: float = Field(
        ..., description="Point estimate of property value"
    )
    confidence_low: float = Field(
        ..., description="Lower bound of 90% confidence interval"
    )
    confidence_high: float = Field(
        ..., description="Upper bound of 90% confidence interval"
    )
    currency: str = Field(default="INR", description="Currency code")
    model_version: str = Field(..., description="Model training timestamp")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


# ── Monitoring Schemas ───────────────────────────────────────────────────

class DriftFeatureDetail(BaseModel):
    psi: float
    ks_statistic: float
    ks_pvalue: float
    status: str


class DriftResponse(BaseModel):
    """Data drift analysis report."""
    status: str = Field(..., description="Overall drift status")
    n_samples: int = Field(0, description="Samples analysed")
    features: dict[str, DriftFeatureDetail] = Field(
        default_factory=dict, description="Per-feature drift metrics"
    )


class MonitoringMetricsResponse(BaseModel):
    """Real-time system metrics."""
    uptime_seconds: float
    total_requests: int
    total_errors: int
    error_rate: float
    requests_per_minute_5m: float
    latency_ms: dict[str, float]
    predictions: dict[str, Any]
    recorded_at: str


class RetrainingCheckDetail(BaseModel):
    passed: bool
    detail: str


class RetrainingResponse(BaseModel):
    """Retraining policy evaluation result."""
    should_retrain: bool
    reasons: list[str]
    model_age_days: float
    checks: dict[str, RetrainingCheckDetail]
    evaluated_at: str


class FeatureVersionSummary(BaseModel):
    version: str
    created_at: str
    n_features: int
    changelog: str
    is_active: bool


class FeatureRegistryResponse(BaseModel):
    """Feature version history."""
    active_version: Optional[str] = None
    versions: list[FeatureVersionSummary]
