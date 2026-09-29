"""
tests/unit/test_contract_compliance_phase4.py
---------------------------------------------
CHECK 1 — Data contract compliance for Phase 4.

Verifies that the Blending Engine takes compliant Phase 1/2 datasets 
and valid Phase 3 weights, and successfully yields a strictly compliant 
blended xr.Dataset that honors the Sacred Data Contract.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.blending.extreme_booster import ExtremeBooster
from src.blending.inverse_rmse_blender import InverseRMSEBlender
from src.ingestion.base_ingestor import BaseIngestor

_LEAD_HOURS = [0, 24, 48, 72, 96, 120]
_REQUIRED_VARS = ["t2m", "tp", "u10", "v10", "mslp"]

class _StubIngestor(BaseIngestor):
    def ingest(self, date, lead_hours):
        pass


@pytest.fixture(scope="class")
def mock_model_datasets() -> dict[str, xr.Dataset]:
    """Creates minimal xr.Datasets for blending, strictly using the data contract."""
    lat = np.arange(6.0, 38.25, 0.25)
    lon = np.arange(68.0, 97.25, 0.25)
    shape = (len(_LEAD_HOURS), len(lat), len(lon))
    
    datasets = {}
    for m_name in ["modelA", "modelB"]:
        data_vars = {
            "t2m": xr.DataArray(np.full(shape, 300.0), dims=["lead_hours", "lat", "lon"]),
            "tp": xr.DataArray(np.full(shape, 0.0), dims=["lead_hours", "lat", "lon"]),
            "u10": xr.DataArray(np.full(shape, 5.0), dims=["lead_hours", "lat", "lon"]),
            "v10": xr.DataArray(np.full(shape, -2.0), dims=["lead_hours", "lat", "lon"]),
            "mslp": xr.DataArray(np.full(shape, 101300.0), dims=["lead_hours", "lat", "lon"]),
        }
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"lead_hours": _LEAD_HOURS, "lat": lat, "lon": lon, "valid_time": pd.Timestamp("2024-01-01")},
            attrs={
                "model_name": m_name,
                "model_type": "nwp",
                "units": "{'t2m': 'K', 'tp': 'mm/hr', 'u10': 'm/s', 'v10': 'm/s', 'mslp': 'Pa'}"
            }
        )
        datasets[m_name] = ds
        
    return datasets


class TestContractCompliancePhase4:
    
    def test_inverse_rmse_blender_output_compliance(self, mock_model_datasets: dict[str, xr.Dataset]) -> None:
        """Verifies InverseRMSEBlender outputs a fully compliant xr.Dataset."""
        # 1. Setup mock requirements
        weights_df = pd.DataFrame(columns=["model", "variable", "region", "lead_hours", "season", "weight"])
        region_masks = {"all_india": np.ones((len(mock_model_datasets["modelA"].coords["lat"]), len(mock_model_datasets["modelA"].coords["lon"])), dtype=bool)}
        meta = {"init_time": pd.Timestamp("2024-01-01"), "season": "DJF", "regime_id": 0}
        
        # 2. Blend
        blender = InverseRMSEBlender(config={})
        blended_ds = blender.blend(mock_model_datasets, weights_df, region_masks, meta)
        
        # 3. Apply Extreme Booster (which adds flags)
        booster = ExtremeBooster(blend_config={"thresholds": {}})
        final_ds = booster.apply(blended_ds, mock_model_datasets, weights_df)
        
        # 4. Assert Output Contract Compliance
        validator = _StubIngestor(config={}, model_config={"name": "hybrid_blend", "type": "ensemble", "key": "hybrid"})
        
        # Will raise ValueError if invalid
        assert validator.validate_output(final_ds) is True
        
        # Assertions mandated by prompt
        assert isinstance(final_ds, xr.Dataset)
        assert set(final_ds.dims) == {"lead_hours", "lat", "lon"}
        assert all(v in final_ds for v in ["t2m", "tp", "u10", "v10", "mslp"])
        assert final_ds.attrs.get("model_name") is not None
        assert final_ds.attrs.get("model_name") == "hybrid_blend"
        
        # Also verify booster extensions
        assert "extreme_flags" in final_ds
        assert "alert_level" in final_ds
