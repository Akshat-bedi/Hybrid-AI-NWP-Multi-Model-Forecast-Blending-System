"""
tests/unit/test_contract_compliance_phase2.py
---------------------------------------------
CHECK 1 — Data contract compliance for Phase 2 preprocessing functions.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pytest
import xarray as xr

from src.ingestion.base_ingestor import BaseIngestor
from src.preprocessing.quality_control import QualityControl
from src.preprocessing.regridder import Regridder
from src.preprocessing.variable_mapper import VariableMapper

_LEAD_HOURS = [0, 24, 48, 72, 96, 120]
_REQUIRED_VARS = ["t2m", "tp", "u10", "v10", "mslp"]


class _StubIngestor(BaseIngestor):
    def ingest(self, date, lead_hours):
        pass


@pytest.fixture(scope="class")
def phase2_result() -> xr.Dataset:
    """Creates a raw dataset, runs it through Phase 2, and returns the result."""
    # 1. Create dataset on non-standard grid with non-standard vars and units
    lat = np.arange(5.0, 39.0, 0.5)
    lon = np.arange(67.0, 98.0, 0.5)
    
    shape = (len(_LEAD_HOURS), len(lat), len(lon))
    
    data_vars = {
        "TMP_2m": xr.DataArray(np.full(shape, 20.0), dims=["lead_hours", "lat", "lon"]),
        "tp": xr.DataArray(np.full(shape, 0.0), dims=["lead_hours", "lat", "lon"]),
        "u10": xr.DataArray(np.full(shape, 5.0), dims=["lead_hours", "lat", "lon"]),
        "v10": xr.DataArray(np.full(shape, -2.0), dims=["lead_hours", "lat", "lon"]),
        "mslp": xr.DataArray(np.full(shape, 101300.0), dims=["lead_hours", "lat", "lon"]),
    }
    
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"lead_hours": _LEAD_HOURS, "lat": lat, "lon": lon},
        attrs={
            "model_name": "test_model",
            "model_type": "nwp",
            "units": "{'TMP_2m': 'C', 'tp': 'kg/m2/s', 'u10': 'm/s', 'v10': 'm/s', 'mslp': 'Pa'}"
        }
    )
    
    model_config = {
        "test_model": {
            "var_mapping": {"TMP_2m": "t2m"},
            "source_units": {"t2m": "C", "tp": "kg/m2/s"}
        }
    }
    
    # 2. Pass it through Phase 2 Pipeline
    ds = VariableMapper().map(ds, "test_model", model_config)
    ds = Regridder().regrid(ds)
    ds = QualityControl().check(ds)
    
    return ds


class TestContractCompliancePhase2:
    
    def test_validate_output_passes(self, phase2_result: xr.Dataset) -> None:
        """3. Asserts output passes BaseIngestor.validate_output()"""
        stub = _StubIngestor(config={}, model_config={"name": "test_model", "type": "nwp", "key": "test_model"})
        assert stub.validate_output(phase2_result) is True
        
    def test_isinstance_xr_dataset(self, phase2_result: xr.Dataset) -> None:
        """4. Assert: isinstance(result, xr.Dataset)"""
        assert isinstance(phase2_result, xr.Dataset)
        
    def test_dimensions(self, phase2_result: xr.Dataset) -> None:
        """5. Assert: set(result.dims) == {'lead_hours', 'lat', 'lon'}"""
        assert set(phase2_result.dims) == {"lead_hours", "lat", "lon"}
        
    def test_variables_present(self, phase2_result: xr.Dataset) -> None:
        """6. Assert: all(v in result for v in ['t2m', 'tp', 'u10', 'v10', 'mslp'])"""
        assert all(v in phase2_result for v in _REQUIRED_VARS)
        
    def test_model_name_present(self, phase2_result: xr.Dataset) -> None:
        """7. Assert: result.attrs.get('model_name') is not None"""
        assert phase2_result.attrs.get("model_name") is not None
