"""
FastAPI Backend -- Property Valuation Service
=============================================
Production-ready API with:
- Input validation via Pydantic
- Real-time inference with pre-loaded model
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
from app.schemas import HealthResponse, PredictionResponse, PropertyInput
from ml.config import LOG_DIR

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


# -- Application Lifecycle --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup."""
    logger.info("Loading model artifacts...")
    try:
        load_artifacts()
        logger.info("Model artifacts loaded -- server ready")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    yield
    logger.info("Shutting down")


# -- App Instance -----------------------------------------------------------
app = FastAPI(
    title="PropertyValuator API",
    description="Real-time residential property price prediction.",
    version="1.0.0",
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
#  Routes
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
    try:
        # 1. Preprocess
        feature_vector = preprocess_input(property_input.model_dump())

        # 2. Predict
        result = predict(feature_vector)

        return PredictionResponse(**result)

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
