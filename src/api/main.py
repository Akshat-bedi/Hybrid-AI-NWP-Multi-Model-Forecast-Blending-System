"""
src/api/main.py
---------------
Layer   : api
Purpose : FastAPI application entrypoint and router registration.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import alerts, forecast, skill, weights
from src.api.schemas import HealthResponse
from src.utils.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="Hybrid Blend API",
    version="1.0.0",
    description="READ-ONLY Data Access API for the Hybrid AI-NWP Blending System."
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Register routers under /api/v1 prefix
app.include_router(forecast.router, prefix="/api/v1")
app.include_router(weights.router, prefix="/api/v1")
app.include_router(skill.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Healthcheck endpoint retrieving system state without computation."""
    try:
        config = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
        models_cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
    except FileNotFoundError:
        return HealthResponse(status="degraded (missing config)", models_loaded=[])

    # Extract enabled models
    models_loaded = [
        key for key, m_cfg in models_cfg.get("models", {}).items()
        if m_cfg.get("enabled", True)
    ]
    
    # Identify the latest run from the blended disk cache
    latest_run = None
    try:
        blended_dir = PROJECT_ROOT / config["paths"]["blended_data"]
        nc_files = list(blended_dir.glob("*_blended.nc"))
        if nc_files:
            latest = max(nc_files)
            latest_run = latest.name.split("_")[0]
    except Exception:
        pass
        
    return HealthResponse(
        status="ok",
        latest_run=latest_run,
        models_loaded=models_loaded
    )
