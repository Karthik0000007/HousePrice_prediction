"""
FastAPI Backend -- Property Valuation Service
=============================================
Production-ready API for a real-estate analytics company, with:
- Input validation via Pydantic
- Real-time inference with pre-loaded model
- Data drift detection (PSI + KS test)
- Prediction monitoring (latency, distribution, error rate)
- Retraining policy evaluation
- Feature version registry
- Health-check endpoint
- Structured JSON responses
- CORS support
- Request logging
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model import load_artifacts, predict, get_metadata
from app.preprocessing import preprocess_input
from app.schemas import (
    DriftResponse,
    FeatureRegistryResponse,
    FeatureVersionSummary,
    HealthResponse,
    MonitoringMetricsResponse,
    PredictionResponse,
    PropertyInput,
    RetrainingResponse,
)
from ml.config import (
    ARTIFACTS_DIR,
    DRIFT_BASELINE_PATH,
    DRIFT_BUFFER_SIZE,
    DRIFT_PSI_BINS,
    FEATURE_REGISTRY_PATH,
    LOG_DIR,
    METADATA_PATH,
    MONITOR_MAX_HISTORY,
)
from ml.drift import DriftDetector, load_baseline
from ml.monitoring import PredictionMonitor
from ml.retraining import RetrainingEvaluator
from ml.feature_registry import FeatureRegistry

# -- Logging ---------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("api")

# -- Module-level singletons (initialised in lifespan) ----------------------
_drift_detector: DriftDetector | None = None
_monitor: PredictionMonitor | None = None
_retraining_eval: RetrainingEvaluator | None = None
_feature_registry: FeatureRegistry | None = None


# -- Application Lifecycle --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts and initialise monitoring subsystems."""
    global _drift_detector, _monitor, _retraining_eval, _feature_registry

    logger.info("Loading model artifacts...")
    try:
        load_artifacts()
        logger.info("Model artifacts loaded -- server ready")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    # Drift detector
    if DRIFT_BASELINE_PATH.exists():
        baseline = load_baseline(DRIFT_BASELINE_PATH)
        meta = get_metadata()
        _drift_detector = DriftDetector(
            reference_data=baseline,
            feature_names=meta.get("feature_names", []),
            n_bins=DRIFT_PSI_BINS,
            buffer_size=DRIFT_BUFFER_SIZE,
        )
    else:
        logger.warning("No drift baseline found; drift detection disabled")

    # Prediction monitor
    _monitor = PredictionMonitor(
        max_history=MONITOR_MAX_HISTORY, log_dir=LOG_DIR
    )

    # Retraining evaluator
    _retraining_eval = RetrainingEvaluator(metadata_path=METADATA_PATH)

    # Feature registry
    _feature_registry = FeatureRegistry(FEATURE_REGISTRY_PATH)

    yield
    # Shutdown: flush metrics
    if _monitor:
        _monitor.flush_to_log()
    logger.info("Shutting down")


# -- App Instance -----------------------------------------------------------
app = FastAPI(
    title="PropertyValuator API",
    description=(
        "Real-time residential property price prediction for "
        "real-estate analytics, with built-in MLOps monitoring."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, images)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -- Middleware: request timing ----------------------------------------------
@app.middleware("http")
async def log_request(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ===========================================================================
#  Core Routes
# ===========================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend UI."""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Check API and model health."""
    meta = get_metadata()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=meta.get("trained_at"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_price(property_input: PropertyInput):
    """
    Predict the market value of a residential property.

    Accepts property features and returns a point estimate
    with a 90% confidence interval.
    """
    start = time.perf_counter()
    try:
        # 1. Preprocess
        feature_vector = preprocess_input(property_input.model_dump())

        # 2. Predict
        result = predict(feature_vector)

        # 3. Record for monitoring
        latency_ms = (time.perf_counter() - start) * 1000
        if _monitor:
            _monitor.record_prediction(result["predicted_price"], latency_ms)
        if _drift_detector:
            _drift_detector.record(feature_vector)

        return PredictionResponse(**result)

    except Exception as e:
        if _monitor:
            _monitor.record_error()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ===========================================================================
#  Monitoring Routes
# ===========================================================================

@app.get(
    "/monitoring/metrics",
    response_model=MonitoringMetricsResponse,
    tags=["monitoring"],
)
async def monitoring_metrics():
    """
    Real-time system metrics: latency percentiles, prediction
    distribution, request volume, error rate, and uptime.
    """
    if not _monitor:
        raise HTTPException(503, "Monitor not initialised")
    return MonitoringMetricsResponse(**_monitor.get_metrics())


@app.get(
    "/monitoring/drift",
    response_model=DriftResponse,
    tags=["monitoring"],
)
async def monitoring_drift(force: bool = False):
    """
    Data drift analysis using PSI and KS tests.

    By default, analysis runs only when the prediction buffer reaches
    the configured threshold.  Pass ``?force=true`` to run immediately.
    """
    if not _drift_detector:
        raise HTTPException(503, "Drift detector not initialised (no baseline)")
    report = _drift_detector.analyze(force=force)
    if report is None:
        return DriftResponse(
            status="insufficient_data",
            n_samples=_drift_detector.buffer_count,
            features={},
        )
    return DriftResponse(**report)


@app.get(
    "/monitoring/retraining",
    response_model=RetrainingResponse,
    tags=["monitoring"],
)
async def monitoring_retraining():
    """
    Evaluate whether the model should be retrained based on:
    - Model age (scheduled cadence)
    - Data drift severity
    - Performance degradation (if labelled feedback available)
    - New data volume
    """
    if not _retraining_eval:
        raise HTTPException(503, "Retraining evaluator not initialised")

    drift_report = None
    if _drift_detector:
        drift_report = _drift_detector.get_last_report()

    result = _retraining_eval.evaluate(drift_report=drift_report)
    return RetrainingResponse(**result)


@app.get(
    "/monitoring/features",
    response_model=FeatureRegistryResponse,
    tags=["monitoring"],
)
async def monitoring_features():
    """
    Feature schema version history and active version.
    """
    if not _feature_registry:
        raise HTTPException(503, "Feature registry not initialised")
    versions = _feature_registry.list_versions()
    active = _feature_registry.get_active()
    return FeatureRegistryResponse(
        active_version=active.version if active else None,
        versions=[FeatureVersionSummary(**v) for v in versions],
    )
