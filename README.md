# PropertyValuator — Real-Time Residential Property Valuation

A production-ready, ML-powered web application that provides **real-time residential property price predictions** with confidence intervals — designed for deployment by a **real-estate analytics company** with built-in MLOps guardrails for data drift detection, model retraining, feature versioning, monitoring, and horizontal scalability.

---

## Problem Statement

Accurate property valuation is critical for buyers, sellers, lenders, and real-estate professionals. Traditional appraisals are slow, subjective, and expensive. **PropertyValuator** automates this by training a machine-learning model on historical housing transaction data and serving instant valuations through a modern web interface.

**Scope:** Single-family residential properties. The model estimates market value (INR) given 12 property attributes covering size, amenities, location preference, and furnishing level.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Browser (UI)                             │
│   index.html  --  vanilla JS  --  responsive CSS                 │
└────────────┬────────────────────┬────────────────────────────────┘
             │  POST /predict     │  GET /health
             v                    v
┌──────────────────────────────────────────────────────────────────┐
│                 FastAPI Backend  (app/main.py)  v2.0             │
│                                                                  │
│  ┌──────────┐  ┌─────────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ Pydantic  │  │Preprocessing│  │  Model   │  │  Monitoring  │ │
│  │Validation │->│(feature eng)│->│Inference │->│   Hooks      │ │
│  │schemas.py │  │preprocess.py│  │ model.py │  │(drift,latency│ │
│  └──────────┘  └─────────────┘  └──────────┘  └──────────────┘ │
│                                                                  │
│  ┌──────────────────────── MLOps Layer ───────────────────────┐  │
│  │ DriftDetector  PredictionMonitor  RetrainingEvaluator      │  │
│  │ (ml/drift.py)  (ml/monitoring.py) (ml/retraining.py)      │  │
│  │                                                            │  │
│  │ FeatureRegistry (ml/feature_registry.py)                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Endpoints: /health  /predict  /monitoring/metrics               │
│             /monitoring/drift  /monitoring/retraining             │
│             /monitoring/features                                 │
└──────────────────────────────────────────────────────────────────┘
                           ^
                           │  python -m ml.train
┌──────────────────────────┴───────────────────────────────────────┐
│                 Training Pipeline  (ml/)                          │
│                                                                  │
│  Housing.csv -> Audit -> Feature Eng -> Split -> Scale           │
│             -> Cross-Val -> GBR Train -> Evaluate -> Save        │
│             -> Save Drift Baseline -> Register Feature Version   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
PropertyValuator/
├── app/                            # Web application
│   ├── main.py                     # FastAPI routes, middleware, lifespan, monitoring endpoints
│   ├── model.py                    # Model loading & inference service
│   ├── preprocessing.py            # Input -> feature vector transform
│   ├── schemas.py                  # Pydantic request/response models (incl. monitoring schemas)
│   └── static/
│       └── index.html              # Single-page frontend (HTML/CSS/JS)
├── ml/                             # Machine learning pipeline + MLOps modules
│   ├── config.py                   # Feature defs, hyperparams, paths, scalability config
│   ├── train.py                    # End-to-end training script (incl. baseline + registry steps)
│   ├── drift.py                    # Data drift detection (PSI + Kolmogorov-Smirnov test)
│   ├── monitoring.py               # Prediction monitoring (latency, volume, error rate)
│   ├── retraining.py               # Policy-driven retraining evaluator
│   └── feature_registry.py         # Semantic-versioned feature schema registry
├── data/
│   └── Housing.csv                 # Source dataset (545 records)
├── artifacts/                      # Generated model artifacts (git-ignored)
│   ├── model.joblib                # Trained GBR model
│   ├── scaler.joblib               # Fitted StandardScaler
│   ├── metadata.json               # Training metadata + metrics
│   ├── drift_baseline.npz          # Per-feature training distributions for drift detection
│   └── feature_registry.json       # Feature version history
├── logs/                           # Runtime logs (git-ignored)
├── Dockerfile                      # Container build
├── docker-compose.yml              # One-command deployment with resource limits
├── requirements.txt                # Python dependencies (incl. scipy)
├── .env                            # Environment config template
├── .gitignore
├── .dockerignore
└── README.md
```

---

## Data & Preprocessing

### Dataset
- **Source:** Housing.csv — 545 residential property records
- **Target:** `price` (INR) — log-transformed during training to reduce skew

### Features

| Feature | Type | Encoding |
|---------|------|----------|
| area | Numeric | StandardScaler |
| bedrooms | Numeric | StandardScaler |
| bathrooms | Numeric | StandardScaler |
| stories | Numeric | StandardScaler |
| parking | Numeric | StandardScaler |
| mainroad | Binary | yes→1, no→0 |
| guestroom | Binary | yes→1, no→0 |
| basement | Binary | yes→1, no→0 |
| hotwaterheating | Binary | yes→1, no→0 |
| airconditioning | Binary | yes→1, no→0 |
| prefarea | Binary | yes→1, no→0 |
| furnishingstatus | Ordinal | unfurnished→0, semi→1, furnished→2 |

### Engineered Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| area_per_bedroom | area / (bedrooms + 1) | Space efficiency per sleeping area |
| area_per_bathroom | area / (bathrooms + 1) | Bathroom-to-space ratio |
| total_rooms | bedrooms + bathrooms | Overall size proxy |
| luxury_score | AC + guestroom + basement + prefarea | Premium amenity index |

### Preprocessing Decisions
- **No missing values** in the dataset (verified at runtime)
- **No duplicates** detected
- **Outlier filtering:** rows outside the 1st–99th percentile on price are removed
- **Scaler fit on training data only** — no data leakage
- **Log1p transform** on target for variance stabilization; `expm1` inverse at inference

---

## Modeling

| Component | Choice |
|-----------|--------|
| Algorithm | `GradientBoostingRegressor` (scikit-learn) |
| Split | 70% train / 15% validation / 15% test |
| Cross-validation | 5-fold CV on training set |
| Metrics | MAE, RMSE, R² (both log-scale and original scale) |
| Confidence interval | 90% CI using test-set residual standard deviation |
| Serialization | `joblib` (model, scaler, feature names) + JSON metadata |

---

## MLOps & Production Operations

This section documents the enterprise-grade operational infrastructure built into PropertyValuator, designed so a real-estate analytics company can run the system in production with confidence.

### 1. Data Drift Detection (`ml/drift.py`)

**Why it matters:** Property markets shift — construction booms, interest rate changes, and urban sprawl alter the statistical profile of incoming requests. A model trained on stale distributions will silently degrade.

**Approach:**

| Method | What It Measures | Threshold |
|--------|------------------|-----------|
| **Population Stability Index (PSI)** | Distribution shift between training baseline and recent requests | < 0.10 stable, 0.10-0.25 moderate, > 0.25 significant |
| **Kolmogorov-Smirnov test** | Maximum distance between empirical CDFs | p-value < 0.05 triggers alert |

- At training time (`ml/train.py` step 12), the **per-feature distributions** of the scaled training set are saved to `artifacts/drift_baseline.npz`.
- At inference time, every prediction request's feature vector is buffered inside `DriftDetector`.
- When the buffer reaches a configurable size (default: 50 samples), PSI and KS statistics are computed against the baseline.
- Results are exposed via `GET /monitoring/drift`, returning per-feature drift status, PSI value, and KS p-value.

**Operational thresholds:**
- **Stable** (PSI < 0.10): No action needed.
- **Moderate** (0.10-0.25): Schedule manual review; potential early retraining.
- **Significant** (PSI > 0.25): Immediate investigation; model predictions may be unreliable.

### 2. Prediction Monitoring (`ml/monitoring.py`)

**Why it matters:** Availability and latency directly impact analyst workflow. Unexpected prediction distributions signal data quality issues or model bugs.

**Tracked metrics:**

| Metric | Details |
|--------|---------|
| Total predictions served | Lifetime counter |
| Latency percentiles | p50, p95, p99 (seconds) |
| Prediction value distribution | mean, min, max, std of predicted prices |
| Error count and rate | HTTP 500 / processing failures |
| Requests per minute (RPM) | 5-minute rolling window |

- `PredictionMonitor` uses a **thread-safe deque** (max 10,000 entries) as a rolling buffer.
- Every `/predict` call records latency + predicted value.
- Every unhandled exception in the prediction path records an error.
- Metrics are available in real time via `GET /monitoring/metrics`.
- `flush_to_log()` writes periodic snapshots to `logs/` for offline analysis.

### 3. Retraining Strategy (`ml/retraining.py`)

**Why it matters:** A model that was accurate at training time will degrade as market conditions change. Reactive retraining (only when users complain) is too late.

**Retraining is evaluated on four independent triggers:**

| Trigger | Policy | Default |
|---------|--------|---------|
| **Scheduled cadence** | Retrain every N days regardless of drift | 30 days |
| **Data drift** | Retrain when >= N features show significant drift (PSI > 0.25) | 2 features |
| **Performance degradation** | Retrain when estimated R² falls below threshold | R² < 0.55 |
| **Data volume** | Retrain when N new labeled samples are available | 100 samples |

**Workflow:**
1. `RetrainingEvaluator.evaluate()` inspects model age, drift state, and monitoring metrics.
2. Returns a structured recommendation: `retrain_recommended` (bool), list of triggered checks with pass/fail, and urgency level (`low` / `medium` / `high`).
3. Results are available via `GET /monitoring/retraining`.
4. **Rollback:** Before any retraining, `archive_current_model()` copies the current `artifacts/` directory to `artifacts/archive/{ISO-timestamp}/`, enabling one-command rollback.

**Shadow evaluation (recommended for production):** Train the new model on the updated dataset, serve both old and new models in parallel, compare A/B metrics for 48+ hours, then promote.

### 4. Feature Versioning (`ml/feature_registry.py`)

**Why it matters:** When you add, remove, or transform a feature, every downstream component (preprocessing, model, schema, drift baselines) must stay in sync. Without versioning, silent schema mismatches cause incorrect predictions.

**Design:**
- A `FeatureRegistry` persists semantic-versioned feature schemas to `artifacts/feature_registry.json`.
- Each `FeatureVersion` records: version string (e.g., `1.0.0`), list of `FeatureDescriptors` (name, dtype, source, transform), creation timestamp, and description.
- The registry supports `register_version()`, `set_active()`, `get_active()`, and `list_versions()`.
- During training (step 13), `build_current_version()` constructs a `FeatureVersion` from `ml/config.py` definitions and registers it.
- The active schema is exposed via `GET /monitoring/features`.

**Versioning convention:**
- **PATCH** (1.0.x): Bug fix in transform logic (no feature add/remove)
- **MINOR** (1.x.0): Add a new engineered feature
- **MAJOR** (x.0.0): Remove a feature or change a feature's dtype/semantics

### 5. Scalability Strategy

**Current architecture (single-node):**
- Docker Compose with resource limits: 512 MB memory, 1 CPU
- Uvicorn with configurable `API_WORKERS` (default: 2, set via `ml/config.py`)
- Model loaded once per worker at startup (no per-request I/O)

**Horizontal scaling path:**

```
                    ┌─────────────┐
                    │   Nginx /   │
                    │ Cloud LB    │
                    └──────┬──────┘
               ┌───────────┼───────────┐
               v           v           v
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │ Worker 1 │ │ Worker 2 │ │ Worker N │
         │ (Gunicorn│ │ (Gunicorn│ │ (Gunicorn│
         │  +Uvicorn│ │  +Uvicorn│ │  +Uvicorn│
         └──────────┘ └──────────┘ └──────────┘
               │           │           │
               └───────────┼───────────┘
                           v
                   ┌──────────────┐
                   │  Shared Vol  │
                   │  (artifacts) │
                   └──────────────┘
```

**To scale with `docker compose`:**
```bash
docker compose up --scale app=4 --build
```
- Add a reverse proxy (nginx / Traefik / cloud ALB) in front.
- Mount `artifacts/` as a shared read-only volume across replicas.
- For monitoring aggregation across replicas, export metrics to Prometheus (future enhancement) or centralize logs.

**Production-grade upgrades (future roadmap):**
- **Kubernetes**: Deploy as a Deployment with HPA (Horizontal Pod Autoscaler) targeting CPU / RPM.
- **Model serving**: Replace custom inference with TorchServe / TF Serving / BentoML for batching, GPU support, and A/B routing.
- **Feature store**: Centralize feature computation with Feast / Tecton to decouple from training pipeline.
- **Database**: Store predictions to PostgreSQL for audit trail and model feedback loop.
- **Message queue**: Use Redis / Kafka for async prediction requests under heavy load.

---

## Quick Start

### Prerequisites
- Python 3.10+ **or** Docker

### Option A — Local Setup

```bash
# 1. Clone & navigate
cd PropertyValuator

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python -m ml.train

# 5. Start the server
uvicorn app.main:app --reload

# 6. Open http://localhost:8000
```

### Option B — Docker

```bash
# Build & run (model trains during build)
docker compose up --build

# Open http://localhost:8000
```

---

## API Reference

### `GET /health`
Returns service health and model metadata.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2026-03-01T12:00:00+00:00"
}
```

### `POST /predict`
Returns predicted price with 90% confidence interval. Also records the prediction in the drift and monitoring subsystems.

**Request:**
```json
{
  "area": 5000,
  "bedrooms": 3,
  "bathrooms": 2,
  "stories": 2,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "parking": 2,
  "prefarea": "yes",
  "furnishingstatus": "semi-furnished"
}
```

**Response:**
```json
{
  "predicted_price": 5425000.00,
  "confidence_low": 4180000.00,
  "confidence_high": 7050000.00,
  "currency": "INR",
  "model_version": "2026-03-01T12:00:00+00:00"
}
```

### `GET /monitoring/metrics`
Returns real-time prediction monitoring statistics.

```json
{
  "total_predictions": 342,
  "total_errors": 1,
  "error_rate": 0.0029,
  "latency_p50_s": 0.0034,
  "latency_p95_s": 0.0087,
  "latency_p99_s": 0.0121,
  "prediction_mean": 6120000.0,
  "prediction_std": 1430000.0,
  "prediction_min": 2100000.0,
  "prediction_max": 14500000.0,
  "rpm_5min": 12.4
}
```

### `GET /monitoring/drift`
Returns per-feature drift analysis (PSI + KS test) comparing recent requests against the training baseline.

```json
{
  "baseline_loaded": true,
  "samples_since_last_analysis": 50,
  "features": [
    {
      "feature": "area",
      "psi": 0.042,
      "ks_statistic": 0.11,
      "ks_p_value": 0.38,
      "status": "stable"
    }
  ]
}
```

### `GET /monitoring/retraining`
Evaluates all retraining triggers and returns a structured recommendation.

```json
{
  "retrain_recommended": false,
  "checks": [
    {"trigger": "scheduled_cadence", "passed": true, "detail": "Model age: 12 days (max 30)"},
    {"trigger": "data_drift", "passed": true, "detail": "0 drifted features (threshold: 2)"},
    {"trigger": "performance_degradation", "passed": true, "detail": "No live R2 available"},
    {"trigger": "data_volume", "passed": true, "detail": "0 new samples (threshold: 100)"}
  ]
}
```

### `GET /monitoring/features`
Returns the active feature schema version and version history.

```json
{
  "active_version": "1.0.0",
  "total_versions": 1,
  "active_schema": {
    "version": "1.0.0",
    "created_at": "2026-03-01T12:00:00",
    "description": "Initial feature set from ml.config",
    "feature_count": 16,
    "features": ["area", "bedrooms", "..."]
  }
}
```

Interactive API docs available at `http://localhost:8000/docs`.

---

## UI Preview

The frontend provides:
- **Structured input form** with labeled fields, placeholders, and range hints
- **Client-side validation** with inline error messages
- **Real-time prediction display** showing estimated price + 90% confidence interval
- **Responsive layout** — works on desktop, tablet, and mobile
- **Minimalistic design** using Inter font, neutral palette, and subtle shadows

---

## Assumptions & Limitations

| Area | Detail |
|------|--------|
| **Dataset size** | 545 records -- sufficient for a demo but a production system would benefit from 10K+ samples |
| **Geography** | Data lacks explicit location/city features; the model cannot account for neighborhood-level variation |
| **Stationarity** | Prices are assumed time-invariant; no temporal features or market-cycle adjustment |
| **Currency** | All values in INR; no multi-currency support |
| **Confidence interval** | Based on Gaussian residual assumption; actual coverage may differ |
| **Single model** | No ensemble of heterogeneous models; GBR was chosen for its balance of performance and interpretability |
| **Security** | No authentication/rate-limiting in the demo; add these before public deployment |
| **Monitoring scope** | In-process monitoring with in-memory buffers; a production deployment should export to Prometheus/Grafana or an APM platform |
| **Drift baseline** | Computed from training set only; for continuous learning, refresh baseline with recent labeled data |
| **Retraining** | Evaluator provides recommendations but does not auto-trigger retraining; an orchestrator (Airflow, Prefect) is needed for full automation |
| **Feature registry** | JSON-file-based; migrate to a database-backed feature store (Feast, Tecton) for multi-team collaboration |

---

## Tech Stack

- **ML:** scikit-learn, pandas, NumPy, scipy
- **Backend:** FastAPI, Uvicorn, Pydantic v2
- **Frontend:** Vanilla HTML/CSS/JS (zero framework dependencies)
- **MLOps:** Custom drift detection (PSI + KS), monitoring, retraining evaluator, feature registry
- **Containerization:** Docker, Docker Compose (with deploy resource limits)
- **Serialization:** joblib, JSON, NumPy npz


