"""
tests/unit/test_extreme_booster.py
----------------------------------
Unit tests for the ExtremeBooster weather signature logic.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.blending.extreme_booster import ExtremeBooster

@pytest.fixture(scope="class")
def mock_blend_datasets() -> tuple[xr.Dataset, dict[str, xr.Dataset]]:
    lat = np.arange(10.0, 11.0, 0.25)
    lon = np.arange(70.0, 71.0, 0.25)
    shape = (1, len(lat), len(lon)) # 4x4
    
    # Blended output with extreme heat and extreme wind in specific pixels
    t2m_arr = np.full(shape, 300.0)
    t2m_arr[0, 0, 0] = 315.0 # Extreme Heat (>40C / 313.15K)
    
    u10_arr = np.full(shape, 5.0)
    u10_arr[0, 1, 1] = 20.0 # Extreme Wind (>15m/s)
    v10_arr = np.full(shape, 0.0)
    
    tp_arr = np.full(shape, 0.0) # Base blended TP is 0
    
    blended_ds = xr.Dataset(
        data_vars={
            "t2m": xr.DataArray(t2m_arr, dims=["lead_hours", "lat", "lon"]),
            "tp": xr.DataArray(tp_arr, dims=["lead_hours", "lat", "lon"]),
            "u10": xr.DataArray(u10_arr, dims=["lead_hours", "lat", "lon"]),
            "v10": xr.DataArray(v10_arr, dims=["lead_hours", "lat", "lon"])
        },
        coords={"lead_hours": [0], "lat": lat, "lon": lon}
    )
    
    # Model A predicts extreme rain in pixel [0, 2, 2]
    tp_model_A = np.full(shape, 0.0)
    tp_model_A[0, 2, 2] = 50.0 # > heavy_rain threshold (~2.68 mm/hr)
    ds_A = xr.Dataset(
        data_vars={"tp": xr.DataArray(tp_model_A, dims=["lead_hours", "lat", "lon"])},
        coords={"lead_hours": [0], "lat": lat, "lon": lon}
    )
    
    # Model B predicts no rain
    ds_B = xr.Dataset(
        data_vars={"tp": xr.DataArray(np.full(shape, 0.0), dims=["lead_hours", "lat", "lon"])},
        coords={"lead_hours": [0], "lat": lat, "lon": lon}
    )
    
    return blended_ds, {"modelA": ds_A, "modelB": ds_B}


class TestExtremeBooster:
    def test_extreme_flags_and_alerts(self, mock_blend_datasets: tuple) -> None:
        blended_ds, model_datasets = mock_blend_datasets
        
        # Skill df giving Model A high weight for TP via ETS
        skill_df = pd.DataFrame([
            {"model": "modelA", "variable": "tp", "region": "all_india", "lead_hours": 0, "season": "DJF", "ets_64.5": 0.9},
            {"model": "modelB", "variable": "tp", "region": "all_india", "lead_hours": 0, "season": "DJF", "ets_64.5": 0.1}
        ])
        
        booster = ExtremeBooster(blend_config={"thresholds": {"heavy_rain_mmday": 64.5, "heatwave_celsius": 40.0, "high_wind_ms": 15.0}})
        
        final_ds = booster.apply(blended_ds, model_datasets, skill_df)
        
        # Verify Extreme Rain re-blending
        # modelA had 50.0 rain. ETS weights are 0.9 and 0.1. Re-blended tp should be 50*0.9 = 45.0
        assert np.isclose(final_ds["tp"].values[0, 2, 2], 45.0)
        
        # Verify Flags
        # flags: 1=rain, 2=heat, 4=wind
        assert final_ds["extreme_flags"].values[0, 0, 0] == 2 # Heat
        assert final_ds["extreme_flags"].values[0, 1, 1] == 4 # Wind
        assert final_ds["extreme_flags"].values[0, 2, 2] == 1 # Rain
        
        # Verify Alerts
        # Heat threshold exceeded -> Level 1
        assert final_ds["alert_level"].values[0, 0, 0] == 1
        # Wind is 20 > 15 (exceeded). 20 < 1.5*15 (22.5) -> Level 1
        assert final_ds["alert_level"].values[0, 1, 1] == 1
        # Rain is 50. Threshold is 2.68. 50 > 2*2.68 -> Level 2 (RED)
        assert final_ds["alert_level"].values[0, 2, 2] == 2
