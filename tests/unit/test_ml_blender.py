"""
tests/unit/test_ml_blender.py
-----------------------------
Unit tests for the XGBoost ML Blender logic.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path

from src.blending.ml_blender import MLBlender


@pytest.fixture(scope="class")
def mock_ml_data() -> tuple[dict[str, list[xr.Dataset]], list[xr.Dataset]]:
    """Creates a minimal archive of forecasts and truth for ML training."""
    lat = np.arange(6.0, 7.0, 0.25)
    lon = np.arange(68.0, 69.0, 0.25)
    lead_hours = [0, 24]
    shape = (len(lead_hours), len(lat), len(lon))
    
    # Truth dataset (Random values around 300K for t2m)
    ds_truth = xr.Dataset(
        data_vars={
            "t2m": xr.DataArray(np.random.normal(300, 2, shape).astype(np.float32), dims=["lead_hours", "lat", "lon"]),
            "tp": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=["lead_hours", "lat", "lon"])
        },
        coords={"lead_hours": lead_hours, "lat": lat, "lon": lon, "valid_time": pd.Timestamp("2024-01-01")},
    )
    
    # Model datasets (Forecasts with some noise added to truth)
    ds_gfs = xr.Dataset(
        data_vars={
            "t2m": xr.DataArray(ds_truth["t2m"].values + np.random.normal(0, 1, shape).astype(np.float32), dims=["lead_hours", "lat", "lon"]),
            "tp": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "u10": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "v10": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "mslp": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=["lead_hours", "lat", "lon"])
        },
        coords={"lead_hours": lead_hours, "lat": lat, "lon": lon, "valid_time": pd.Timestamp("2024-01-01")},
    )
    
    ds_ecmwf = xr.Dataset(
        data_vars={
            "t2m": xr.DataArray(ds_truth["t2m"].values + np.random.normal(0, 0.5, shape).astype(np.float32), dims=["lead_hours", "lat", "lon"]),
            "tp": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "u10": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "v10": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=["lead_hours", "lat", "lon"]),
            "mslp": xr.DataArray(np.zeros(shape, dtype=np.float32), dims=["lead_hours", "lat", "lon"])
        },
        coords={"lead_hours": lead_hours, "lat": lat, "lon": lon, "valid_time": pd.Timestamp("2024-01-01")},
    )

    forecast_archive = {
        "gfs": [ds_gfs, ds_gfs],       # 2 samples
        "ecmwf": [ds_ecmwf, ds_ecmwf]
    }
    truth_archive = [ds_truth, ds_truth]
    
    return forecast_archive, truth_archive


class TestMLBlender:
    def test_train_and_predict(self, mock_ml_data: tuple, tmp_path: Path) -> None:
        """Tests that ML Blender can successfully build features, train, and save to disk."""
        forecast_archive, truth_archive = mock_ml_data
        
        blender = MLBlender(config={}, model_dir=str(tmp_path))
        
        # 1. Train the model for 't2m'
        blender.train(forecast_archive, truth_archive, variable="t2m")
        
        # Assert model is saved in memory and on disk
        assert "t2m" in blender.models
        assert (tmp_path / "ml_blender_t2m.joblib").exists()
        
        # 2. Test blending inference
        # Convert archive format to inference format dict[str, xr.Dataset]
        model_datasets = {
            "gfs": forecast_archive["gfs"][0],
            "ecmwf": forecast_archive["ecmwf"][0]
        }
        
        # For inference, it expects datasets with all required validation vars.
        # We'll just patch the InverseRMSEBlender fallback so we don't need a massive mock.
        blender._VARIABLES = ["t2m"]
        
        region_masks = {"all_india": np.ones_like(forecast_archive["gfs"][0]["t2m"].values[0], dtype=bool)}
        meta = {"init_time": pd.Timestamp("2024-01-01")}
        
        # We need an empty weights df for the InverseRMSE fallback which is triggered for un-trained vars
        weights = pd.DataFrame(columns=["model", "variable", "region", "lead_hours", "season", "weight"])
        
        # In this mock, we only trained "t2m". If we pass the full fallback to base it will crash without u10/v10
        # So we mock out InverseRMSEBlender entirely.
        import unittest.mock as mock
        with mock.patch("src.blending.ml_blender.InverseRMSEBlender.blend") as mock_fallback:
            mock_fallback.return_value = model_datasets["gfs"]  # Dummy return
            
            blended = blender.blend(model_datasets, weights, region_masks, meta)
            
            # Assert inference succeeded for the ML trained variable
            assert "t2m" in blended.data_vars
            assert blended["t2m"].shape == forecast_archive["gfs"][0]["t2m"].shape
            assert blended.attrs["blend_method"] == "ml_xgboost"
