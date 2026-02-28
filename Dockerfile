# ── Build Stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ml/ ml/
COPY app/ app/
COPY data/ data/

# Train model at build time so the image ships with artifacts
RUN python -m ml.train

# ── Runtime ──────────────────────────────────────────────────────────────
EXPOSE 8000

# Run with uvicorn — single worker sufficient for demo; scale with gunicorn in prod
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
