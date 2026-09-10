"""
Algeria Wildfire Risk API.

Run from the project root with:
    python -m uvicorn backend.app.main:app --reload --port 8000
"""

import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import CommunePrediction, CommuneUnavailable, DailySummary
from .settings import settings
from .store import (
    get_commune_lookup,
    get_geojson,
    get_historical,
    get_priority,
    get_run_status,
    get_summary,
    get_wilaya_stats,
)


app = FastAPI(
    title="Algeria Wildfire Risk API",
    description="Next-day commune-level wildfire risk predictions for Algeria",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@app.get("/api/run-status")
def run_status():
    """Pipeline freshness, health, and model metadata for the application chrome."""
    return get_run_status()


@app.get("/api/predict/daily")
def daily_geojson():
    """Full national GeoJSON for the map choropleth."""
    return get_geojson()


@app.get("/api/predict/summary", response_model=DailySummary)
def daily_summary():
    """Risk counts and data quality status for the overview."""
    return get_summary()


@app.get("/api/predict/history")
def prediction_history(commune_id: str | None = None, limit: int = 30):
    """Historical national or commune-level series from dated prediction archives."""
    return get_historical(commune_id=commune_id, limit=limit)


@app.get("/api/predict/priority")
def priority_list(limit: int = 20):
    """Top N highest-urgency communes."""
    return get_priority(limit)


@app.get("/api/predict/wilayas")
def wilaya_stats():
    """Risk aggregation per wilaya."""
    return get_wilaya_stats()


@app.get("/api/predict/{commune_id}", response_model=CommunePrediction | CommuneUnavailable)
def commune_detail(commune_id: str):
    """Return prediction detail or a rigorous no-current-prediction explanation."""
    return get_commune_lookup(commune_id)
