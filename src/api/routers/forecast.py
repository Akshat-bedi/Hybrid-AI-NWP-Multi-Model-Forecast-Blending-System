"""
src/api/routers/forecast.py
---------------------------
Layer   : api
Purpose : Endpoints for serving GeoJSON forecast representations.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

import ast
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from fastapi import APIRouter, HTTPException

from src.api.schemas import ForecastResponse, GeoJSONFeature
from src.utils.config_loader import load_config

router = APIRouter(prefix="/forecast", tags=["Forecast"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@router.get("/{variable}/{lead_hours}", response_model=ForecastResponse)
def get_forecast(variable: str, lead_hours: int, init_time: Optional[str] = None):
    """Retrieve 0.5° downsampled GeoJSON grid representations of variables."""
    valid_vars = {"t2m", "tp", "u10", "v10", "mslp", "wind_speed", "alert_level"}
    if variable not in valid_vars:
        raise HTTPException(status_code=404, detail=f"Variable not found. Must be one of {valid_vars}")

    try:
        config = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
        blended_dir = PROJECT_ROOT / config["paths"]["blended_data"]
        
        if init_time:
            file_path = blended_dir / f"{init_time}_blended.nc"
            if not file_path.exists():
                raise FileNotFoundError()
        else:
            nc_files = list(blended_dir.glob("*_blended.nc"))
            if not nc_files:
                raise FileNotFoundError()
            file_path = max(nc_files)
            
    except Exception:
        raise HTTPException(status_code=404, detail="No forecast available")

    try:
        with xr.open_dataset(file_path) as ds:
            if variable not in ds.data_vars:
                raise HTTPException(status_code=404, detail=f"Variable {variable} missing")
                
            if lead_hours not in ds.coords["lead_hours"].values:
                raise HTTPException(status_code=404, detail=f"Lead hour {lead_hours} missing")
                
            sliced = ds[variable].sel(lead_hours=lead_hours)
            
            # Downsample to 0.5° resolution (slice every 2nd point)
            sliced = sliced.isel(lat=slice(None, None, 2), lon=slice(None, None, 2))
            
            # Safely extract units
            unit_str = ds.attrs.get("units", "{}")
            var_unit = "unknown"
            try:
                if isinstance(unit_str, str):
                    unit_dict = ast.literal_eval(unit_str)
                    var_unit = unit_dict.get(variable, "unknown")
            except Exception:
                pass
                
            lats = sliced.coords["lat"].values
            lons = sliced.coords["lon"].values
            vals = sliced.values
            
            features = []
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    val = vals[i, j]
                    if not np.isnan(val):
                        features.append(
                            GeoJSONFeature(
                                geometry={"type": "Point", "coordinates": [float(lon), float(lat)]},
                                properties={"value": float(val), "unit": var_unit, "lead_hours": lead_hours}
                            )
                        )
                        
            derived_init_time = file_path.name.split("_")[0]
            
            return ForecastResponse(
                init_time=derived_init_time,
                variable=variable,
                lead_hours=lead_hours,
                unit=var_unit,
                n_features=len(features),
                features=features
            )
            
    except HTTPException:
        raise
    except Exception:
        # Zero computation failing safely
        raise HTTPException(status_code=404, detail="No forecast available or parsing failed")
