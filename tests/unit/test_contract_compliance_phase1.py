"""
tests/unit/test_contract_compliance_phase1.py
----------------------------------------------
CHECK 1 — Data contract compliance for Phase 1 ingestion functions.

Exercises the exact 7 assertions mandated by the check:
  1. Loads / creates a minimal xr.Dataset using the data contract
  2. Passes it through BaseIngestor.validate_output() and DemoIngestor.ingest()
  3. Asserts validate_output() passes
  4. isinstance(result, xr.Dataset)
  5. set(result.dims) == {"lead_hours", "lat", "lon"}
  6. all(v in result for v in ["t2m", "tp", "u10", "v10", "mslp"])
  7. result.attrs.get("model_name") is not None

Two test classes:
  - TestContractViaValidateOutput  : in-memory dataset, exercises validate_output directly
  - TestContractViaDemoIngest      : writes a real NetCDF, exercises DemoIngestor.ingest()

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.base_ingestor import BaseIngestor  # noqa: E402
from src.ingestion.demo_ingestor import DemoIngestor  # noqa: E402

# ---------------------------------------------------------------------------
# Shared contract constants (Sacred Data Contract)
# ---------------------------------------------------------------------------

_LEAD_HOURS: list[int] = [0, 24, 48, 72, 96, 120]
_LAT: np.ndarray = np.arange(6.0, 38.25, 0.25)
_LON: np.ndarray = np.arange(68.0, 97.25, 0.25)
_SHAPE: tuple[int, ...] = (len(_LEAD_HOURS), len(_LAT), len(_LON))
_COORDS: dict = {"lead_hours": _LEAD_HOURS, "lat": _LAT, "lon": _LON}
_REQUIRED_VARS: list[str] = ["t2m", "tp", "u10", "v10", "mslp"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_dataset(
    model_name: str = "test_model",
    model_type: str = "nwp",
) -> xr.Dataset:
    """Build the smallest possible contract-compliant xr.Dataset in memory."""
    data_vars = {
        var: xr.DataArray(
            np.ones(_SHAPE, dtype=np.float32),
            dims=["lead_hours", "lat", "lon"],
            coords=_COORDS,
        )
        for var in _REQUIRED_VARS
    }
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            "model_name": model_name,
            "model_type": model_type,
            "units": str({"t2m": "K", "tp": "mm/hr", "u10": "m/s",
                          "v10": "m/s", "mslp": "Pa"}),
        },
    )


class _StubIngestor(BaseIngestor):
    """Minimal concrete subclass — allows instantiating the ABC in tests."""

    def ingest(self, date: pd.Timestamp, lead_hours: list[int]) -> xr.Dataset:  # noqa: D102
        raise NotImplementedError("stub only")


# ===========================================================================
# CHECK 1 — Path A: in-memory dataset through BaseIngestor.validate_output()
# ===========================================================================

class TestContractViaValidateOutput:
    """Creates a minimal dataset and passes it through validate_output().

    This tests the function directly without any I/O.
    """

    @pytest.fixture(scope="class")
    def ingestor(self) -> _StubIngestor:
        """Concrete ingestor wired with minimal configs."""
        return _StubIngestor(
            config={"paths": {"raw_data": "data/raw"}},
            model_config={"name": "test_model", "type": "nwp", "key": "test"},
        )

    @pytest.fixture(scope="class")
    def result(self, ingestor: _StubIngestor) -> xr.Dataset:
        """Build dataset and run it through validate_output — this IS the SUT call."""
        ds = _make_minimal_dataset()
        ingestor.validate_output(ds)   # Must not raise; this exercises the Phase 1 code
        return ds

    # --- Assertion 3 -------------------------------------------------------
    def test_validate_output_passes(
        self, ingestor: _StubIngestor, result: xr.Dataset
    ) -> None:
        """Assertion 3: validate_output() must return True."""
        assert ingestor.validate_output(result) is True

    # --- Assertion 4 -------------------------------------------------------
    def test_isinstance_xr_dataset(self, result: xr.Dataset) -> None:
        """Assertion 4: result must be an xr.Dataset."""
        assert isinstance(result, xr.Dataset)

    # --- Assertion 5 -------------------------------------------------------
    def test_dimensions(self, result: xr.Dataset) -> None:
        """Assertion 5: exactly the three required dimensions."""
        assert set(result.dims) == {"lead_hours", "lat", "lon"}

    # --- Assertion 6 -------------------------------------------------------
    def test_all_required_variables_present(self, result: xr.Dataset) -> None:
        """Assertion 6: all five contract variables must be present."""
        assert all(v in result for v in _REQUIRED_VARS)

    # --- Assertion 7 -------------------------------------------------------
    def test_model_name_attr_not_none(self, result: xr.Dataset) -> None:
        """Assertion 7: model_name attribute must be present and non-None."""
        assert result.attrs.get("model_name") is not None

    # --- Extra coord checks ------------------------------------------------
    def test_lead_hours_exact_values(self, result: xr.Dataset) -> None:
        np.testing.assert_array_equal(
            result.coords["lead_hours"].values, _LEAD_HOURS
        )

    def test_lat_in_india_domain(self, result: xr.Dataset) -> None:
        assert 5.5 <= float(result.coords["lat"].min()) <= 6.5
        assert 37.5 <= float(result.coords["lat"].max()) <= 38.5

    def test_lon_in_india_domain(self, result: xr.Dataset) -> None:
        assert 67.5 <= float(result.coords["lon"].min()) <= 68.5
        assert 96.5 <= float(result.coords["lon"].max()) <= 97.5


# ===========================================================================
# CHECK 1 — Path B: NetCDF on disk through DemoIngestor.ingest()
# ===========================================================================

class TestContractViaDemoIngest:
    """Writes a contract-compliant NetCDF to tmp_path, then calls
    DemoIngestor.ingest() and asserts all 7 contract properties.
    """

    _MODEL_KEY = "pangu"
    _DATE = pd.Timestamp("2024-06-01")

    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory: pytest.TempPathFactory) -> xr.Dataset:
        """Full pipeline fixture: write NC → ingest() → return dataset."""
        raw_root = tmp_path_factory.mktemp("demo_raw")
        nc_path = (
            raw_root / self._MODEL_KEY / f"{self._DATE.strftime('%Y%m%d')}.nc"
        )
        nc_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a contract-compliant NetCDF to disk
        _make_minimal_dataset(
            model_name=self._MODEL_KEY, model_type="ai"
        ).to_netcdf(nc_path)

        # Construct ingestor with config pointing at our tmp directory
        ingestor = DemoIngestor(
            config={"paths": {"raw_data": str(raw_root)}},
            model_config={
                "name": "Pangu-Weather",
                "type": "ai",
                "source": "demo",
                "key": self._MODEL_KEY,
            },
        )
        # This is the Phase 1 function under test
        return ingestor.ingest(self._DATE)

    # --- Assertion 3 -------------------------------------------------------
    def test_validate_output_passes(self, result: xr.Dataset) -> None:
        """Assertion 3: a freshly constructed ingestor must accept the result."""
        stub = _StubIngestor(
            {"paths": {"raw_data": "data/raw"}},
            {"name": "test", "type": "ai", "key": "test"},
        )
        assert stub.validate_output(result) is True

    # --- Assertion 4 -------------------------------------------------------
    def test_isinstance_xr_dataset(self, result: xr.Dataset) -> None:
        """Assertion 4: result must be an xr.Dataset."""
        assert isinstance(result, xr.Dataset)

    # --- Assertion 5 -------------------------------------------------------
    def test_dimensions(self, result: xr.Dataset) -> None:
        """Assertion 5: exactly {"lead_hours", "lat", "lon"}."""
        assert set(result.dims) == {"lead_hours", "lat", "lon"}

    # --- Assertion 6 -------------------------------------------------------
    def test_all_required_variables_present(self, result: xr.Dataset) -> None:
        """Assertion 6: all five contract variables present."""
        assert all(v in result for v in _REQUIRED_VARS)

    # --- Assertion 7 -------------------------------------------------------
    def test_model_name_attr_not_none(self, result: xr.Dataset) -> None:
        """Assertion 7: model_name attribute present and non-None."""
        assert result.attrs.get("model_name") is not None

    # --- Additional correctness checks -------------------------------------
    def test_model_name_is_pangu(self, result: xr.Dataset) -> None:
        """model_name must match what was written to the NetCDF."""
        assert result.attrs["model_name"] == self._MODEL_KEY

    def test_model_type_is_ai(self, result: xr.Dataset) -> None:
        """model_type must be 'ai' for Pangu-Weather."""
        assert result.attrs["model_type"] == "ai"

    def test_lead_hours_exact_values(self, result: xr.Dataset) -> None:
        np.testing.assert_array_equal(
            result.coords["lead_hours"].values, _LEAD_HOURS
        )
