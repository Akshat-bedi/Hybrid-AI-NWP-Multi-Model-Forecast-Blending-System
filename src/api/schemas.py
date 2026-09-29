"""
src/api/schemas.py
------------------
Layer   : api
Purpose : Pydantic v2 schemas defining API response shapes and data validation.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Payload for API health checks."""
    status: str
    latest_run: Optional[str] = None
    models_loaded: List[str]


class GeoJSONFeature(BaseModel):
    """Standard GeoJSON Feature representation for individual grid points."""
    type: str = "Feature"
    geometry: dict
    properties: dict


class ForecastResponse(BaseModel):
    """Payload returning a flattened GeoJSON array of forecast values."""
    init_time: str
    variable: str
    lead_hours: int
    unit: str
    n_features: int
    features: List[GeoJSONFeature]


class WeightEntry(BaseModel):
    """Model weight representation for a specific region slice."""
    model_name: str
    weight: float
    rmse: float
    color: str
    skill_source: str = "live"


class WeightMapResponse(BaseModel):
    """Payload mapping regions to their optimized model weight distribution."""
    variable: str
    lead_hours: int
    season: str
    regions: Dict[str, List[WeightEntry]]


class SkillRow(BaseModel):
    """Flattened skill metric scores for a model."""
    model: str
    variable: str
    region: str
    lead_hours: int
    rmse: float
    mae: float
    bias: float
    ets: float
    skill_source: str = "live"


class AlertItem(BaseModel):
    """Severe weather alert detected by the ExtremeBooster."""
    alert_type: str
    severity: int
    region: str
    valid_time: str
    lat_center: float
    lon_center: float
    description: str
