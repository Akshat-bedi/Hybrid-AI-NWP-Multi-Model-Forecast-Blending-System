"""
tests/unit/test_inverse_rmse_blender.py
---------------------------------------
Unit tests for the InverseRMSEBlender fallback logic and computation.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.blending.inverse_rmse_blender import InverseRMSEBlender


@pytest.fixture(scope="class")
def mock_datasets() -> dict[str, xr.Dataset]:
    lat = np.arange(6.0, 38.25, 0.25)
    lon = np.arange(68.0, 97.25, 0.25)
    lead_hours = [0, 24, 48, 72, 96, 120]
    shape = (len(lead_hours), len(lat), len(lon))
    
    ds_A = xr.Dataset(
        data_vars={
            "t2m": xr.DataArray(np.full(shape, 300.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "tp": xr.DataArray(np.full(shape, 10.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "u10": xr.DataArray(np.full(shape, 5.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "v10": xr.DataArray(np.full(shape, 5.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "mslp": xr.DataArray(np.full(shape, 100000.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"])
        },
        coords={"lead_hours": lead_hours, "lat": lat, "lon": lon},
        attrs={"model_name": "modelA", "model_type": "nwp"}
    )
    
    ds_B = xr.Dataset(
        data_vars={
            "t2m": xr.DataArray(np.full(shape, 310.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "tp": xr.DataArray(np.full(shape, 20.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "u10": xr.DataArray(np.full(shape, 5.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "v10": xr.DataArray(np.full(shape, 5.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "mslp": xr.DataArray(np.full(shape, 100000.0, dtype=np.float32), dims=["lead_hours", "lat", "lon"])
        },
        coords={"lead_hours": lead_hours, "lat": lat, "lon": lon},
        attrs={"model_name": "modelB", "model_type": "nwp"}
    )
    
    return {"modelA": ds_A, "modelB": ds_B}


class TestInverseRMSEBlender:
    def test_blend_with_valid_weights(self, mock_datasets: dict[str, xr.Dataset]) -> None:
        """Tests that blending works correctly when exact weights are provided."""
        weights_df = pd.DataFrame([
            {"model": "modelA", "variable": "t2m", "region": "all_india", "lead_hours": 0, "season": "DJF", "weight": 0.8},
            {"model": "modelB", "variable": "t2m", "region": "all_india", "lead_hours": 0, "season": "DJF", "weight": 0.2},
            {"model": "modelA", "variable": "tp", "region": "all_india", "lead_hours": 0, "season": "DJF", "weight": 0.5},
            {"model": "modelB", "variable": "tp", "region": "all_india", "lead_hours": 0, "season": "DJF", "weight": 0.5}
        ])
        
        region_masks = {"all_india": np.ones((129, 117), dtype=bool)}
        meta = {"init_time": pd.Timestamp("2024-01-01"), "season": "DJF", "regime_id": 0}
        
        # Override not needed, datasets now have all vars
        blender = InverseRMSEBlender(config={})
        
        blended = blender.blend(mock_datasets, weights_df, region_masks, meta)
        
        # t2m should be 300 * 0.8 + 310 * 0.2 = 240 + 62 = 302
        assert np.allclose(blended["t2m"].values[0], 302.0)
        
        # tp should be 10 * 0.5 + 20 * 0.5 = 15
        assert np.allclose(blended["tp"].values[0], 15.0)

    def test_fallback_to_equal_weights(self, mock_datasets: dict[str, xr.Dataset]) -> None:
        """Tests the blender's failsafe when weights are entirely missing."""
        weights_df = pd.DataFrame(columns=["model", "variable", "region", "lead_hours", "season", "weight"])
        region_masks = {"all_india": np.ones((129, 117), dtype=bool)}
        meta = {"init_time": pd.Timestamp("2024-01-01"), "season": "DJF", "regime_id": 0}
        
        blender = InverseRMSEBlender(config={})
        
        blended = blender.blend(mock_datasets, weights_df, region_masks, meta)
        
        # With missing weights, it should fall back to 50/50 equal weights
        # t2m = (300 + 310) / 2 = 305
        assert np.allclose(blended["t2m"].values, 305.0)
        # tp = (10 + 20) / 2 = 15
        assert np.allclose(blended["tp"].values, 15.0)
