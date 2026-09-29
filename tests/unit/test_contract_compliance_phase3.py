"""
tests/unit/test_contract_compliance_phase3.py
---------------------------------------------
CHECK 1 — Data contract compliance for Phase 3.

Note: The "Sacred Data Contract" mandates that all *internal preprocessing 
and ingestion* functions return an xr.Dataset. However, Phase 3 (Skill) is explicitly 
commanded by the phase rules to output a pd.DataFrame with specific columns. 

This test verifies that when providing fully compliant xr.Dataset inputs 
(from Phase 1 & 2), the Phase 3 Engine processes them without errors and 
outputs a strictly compliant pd.DataFrame as per the Phase 3 specifications.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.ingestion.base_ingestor import BaseIngestor
from src.skill.skill_computer import SkillComputer
from src.skill.weight_generator import compute_weights

_LEAD_HOURS = [0, 24, 48, 72, 96, 120]
_REQUIRED_VARS = ["t2m", "tp", "u10", "v10", "mslp"]

class _StubIngestor(BaseIngestor):
    def ingest(self, date, lead_hours):
        pass


@pytest.fixture(scope="class")
def compliant_xr_dataset() -> xr.Dataset:
    """Creates a minimal xr.Dataset strictly using the data contract."""
    lat = np.arange(6.0, 38.25, 0.25)
    lon = np.arange(68.0, 97.25, 0.25)
    
    shape = (len(_LEAD_HOURS), len(lat), len(lon))
    
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
            "model_name": "test_nwp",
            "model_type": "nwp",
            "units": "{'t2m': 'K', 'tp': 'mm/hr', 'u10': 'm/s', 'v10': 'm/s', 'mslp': 'Pa'}"
        }
    )
    return ds


class TestContractCompliancePhase3:
    
    def test_input_dataset_is_strictly_compliant(self, compliant_xr_dataset: xr.Dataset) -> None:
        """Verify the mock data entering Phase 3 passes BaseIngestor.validate_output()"""
        stub = _StubIngestor(config={}, model_config={"name": "test_nwp", "type": "nwp", "key": "test_nwp"})
        assert stub.validate_output(compliant_xr_dataset) is True
        
    def test_phase3_output_dataframe_compliance(self, compliant_xr_dataset: xr.Dataset) -> None:
        """Verifies Phase 3 consumes the xr.Dataset and outputs the required pd.DataFrame."""
        
        # 1. Setup Phase 3 Orchestrator
        regions_config = {
            "regions": {
                "test_region": {"lat_min": 10.0, "lat_max": 20.0, "lon_min": 70.0, "lon_max": 80.0}
            }
        }
        computer = SkillComputer(config={}, regions=regions_config)
        
        # 2. Pass compliant xr.Datasets into Phase 3
        forecast_archive = {"test_nwp": [compliant_xr_dataset]}
        truth_archive = [compliant_xr_dataset] # Perfect forecast
        
        skill_df = computer.compute(forecast_archive, truth_archive)
        final_df = compute_weights(skill_df, method="inverse_rmse")
        
        # 3. Assert Output Type
        # Phase 3 specifically dictates returning pd.DataFrame, not xr.Dataset!
        assert isinstance(final_df, pd.DataFrame)
        
        # 4. Assert all required Phase 3 DataFrame columns exist
        expected_cols = {"model", "variable", "region", "lead_hours", "season", 
                         "rmse", "mae", "bias", "ets_64.5", "weight"}
        assert expected_cols.issubset(set(final_df.columns))
        
        # 5. Assert valid rows generated across the variables and lead hours
        assert len(final_df) > 0
        
        # 6. Assert weights normalize exactly to 1.0 per group (the core rule of Phase 3)
        group_sums = final_df.groupby(["variable", "region", "lead_hours", "season"])["weight"].sum()
        assert np.allclose(group_sums.values, 1.0)
