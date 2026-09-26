"""
tests/unit/test_base_ingestor.py
--------------------------------
Unit tests for BaseIngestor.validate_output() and the custom exceptions.

A minimal concrete subclass (_StubIngestor) is defined locally to
instantiate the ABC without implementing real I/O.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.base_ingestor import (  # noqa: E402
    BaseIngestor,
    DataContractError,
    IngestionError,
)

# ---------------------------------------------------------------------------
# Contract-compliant test helpers
# ---------------------------------------------------------------------------

_LEAD_HOURS = [0, 24, 48, 72, 96, 120]
_LAT = np.arange(6.0, 38.25, 0.25)   # 129 pts
_LON = np.arange(68.0, 97.25, 0.25)  # 117 pts
_SHAPE = (len(_LEAD_HOURS), len(_LAT), len(_LON))
_COORDS = {"lead_hours": _LEAD_HOURS, "lat": _LAT, "lon": _LON}
_VALID_ATTRS = {"model_name": "test_model", "model_type": "nwp", "units": "{}"}


def _make_valid_dataset() -> xr.Dataset:
    """Return a minimal contract-compliant xr.Dataset."""
    data_vars = {
        var: xr.DataArray(
            np.ones(_SHAPE, dtype=np.float32),
            dims=["lead_hours", "lat", "lon"],
            coords=_COORDS,
        )
        for var in BaseIngestor.REQUIRED_VARS
    }
    return xr.Dataset(data_vars=data_vars, attrs=dict(_VALID_ATTRS))


# ---------------------------------------------------------------------------
# Concrete stub — only the abstract method matters for these tests
# ---------------------------------------------------------------------------

class _StubIngestor(BaseIngestor):
    """Minimal concrete subclass for testing BaseIngestor."""

    def ingest(self, date: pd.Timestamp, lead_hours: list[int]) -> xr.Dataset:  # noqa: D102
        raise NotImplementedError("stub only")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_ingestor() -> _StubIngestor:
    """Return a _StubIngestor with minimal config dicts."""
    config = {"paths": {"raw_data": "data/raw"}}
    model_config = {"name": "test_model", "type": "nwp", "key": "test"}
    return _StubIngestor(config, model_config)


@pytest.fixture
def valid_ds() -> xr.Dataset:
    """Return a fresh contract-compliant xr.Dataset per test."""
    return _make_valid_dataset()


# ===========================================================================
# Exceptions are importable and inherit from the right base
# ===========================================================================

class TestCustomExceptions:
    def test_ingestion_error_is_exception(self) -> None:
        assert issubclass(IngestionError, Exception)

    def test_data_contract_error_is_exception(self) -> None:
        assert issubclass(DataContractError, Exception)

    def test_ingestion_error_carries_message(self) -> None:
        exc = IngestionError("file missing: /some/path.nc")
        assert "file missing" in str(exc)

    def test_data_contract_error_carries_message(self) -> None:
        exc = DataContractError("Missing required variables: ['tp']")
        assert "tp" in str(exc)


# ===========================================================================
# BaseIngestor.__init__
# ===========================================================================

class TestBaseIngestorInit:
    def test_config_is_stored(self, stub_ingestor: _StubIngestor) -> None:
        assert "paths" in stub_ingestor.config

    def test_model_config_is_stored(self, stub_ingestor: _StubIngestor) -> None:
        assert stub_ingestor.model_config["key"] == "test"

    def test_is_abstract(self) -> None:
        """BaseIngestor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseIngestor(  # type: ignore[abstract]
                {"paths": {}}, {"name": "x", "type": "nwp"}
            )


# ===========================================================================
# validate_output — positive path
# ===========================================================================

class TestValidateOutputPositive:
    def test_returns_true_for_valid_dataset(
        self, stub_ingestor: _StubIngestor, valid_ds: xr.Dataset
    ) -> None:
        assert stub_ingestor.validate_output(valid_ds) is True

    def test_accepts_ai_model_type(
        self, stub_ingestor: _StubIngestor, valid_ds: xr.Dataset
    ) -> None:
        ds = valid_ds.assign_attrs(model_type="ai")
        assert stub_ingestor.validate_output(ds) is True

    def test_accepts_ensemble_model_type(
        self, stub_ingestor: _StubIngestor, valid_ds: xr.Dataset
    ) -> None:
        ds = valid_ds.assign_attrs(model_type="ensemble")
        assert stub_ingestor.validate_output(ds) is True

    def test_warns_on_all_nan_slice_but_returns_true(
        self,
        stub_ingestor: _StubIngestor,
        valid_ds: xr.Dataset,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """All-NaN lead-hour slice must warn, not raise."""
        nan_vals = valid_ds["t2m"].values.copy()
        nan_vals[0] = np.nan  # Entire first lead-hour slice → NaN
        ds = valid_ds.copy()
        ds["t2m"] = xr.DataArray(
            nan_vals, dims=["lead_hours", "lat", "lon"], coords=_COORDS
        )
        with caplog.at_level(logging.WARNING):
            result = stub_ingestor.validate_output(ds)
        assert result is True
        assert any("All-NaN" in r.message for r in caplog.records)


# ===========================================================================
# validate_output — negative path (one test per contract rule)
# ===========================================================================

class TestValidateOutputNegative:
    def test_rejects_non_dataset(self, stub_ingestor: _StubIngestor) -> None:
        with pytest.raises(DataContractError, match="xr.Dataset"):
            stub_ingestor.validate_output({"t2m": 1})  # type: ignore[arg-type]

    def test_rejects_dataframe(self, stub_ingestor: _StubIngestor) -> None:
        import pandas as pd
        with pytest.raises(DataContractError, match="xr.Dataset"):
            stub_ingestor.validate_output(pd.DataFrame())  # type: ignore[arg-type]

    def test_rejects_missing_variable(
        self, stub_ingestor: _StubIngestor, valid_ds: xr.Dataset
    ) -> None:
        broken = valid_ds.drop_vars("tp")
        with pytest.raises(DataContractError, match="Missing required variables"):
            stub_ingestor.validate_output(broken)

    def test_rejects_wrong_dimensions(self, stub_ingestor: _StubIngestor) -> None:
        bad = xr.Dataset(
            {
                var: xr.DataArray(np.zeros((3, 3)), dims=["x", "y"])
                for var in BaseIngestor.REQUIRED_VARS
            },
            attrs=_VALID_ATTRS,
        )
        with pytest.raises(DataContractError, match="Dimension mismatch"):
            stub_ingestor.validate_output(bad)

    def test_rejects_wrong_lead_hours(
        self, stub_ingestor: _StubIngestor, valid_ds: xr.Dataset
    ) -> None:
        broken = valid_ds.assign_coords(lead_hours=[0, 6, 12, 18, 24, 30])
        with pytest.raises(DataContractError, match="lead_hours"):
            stub_ingestor.validate_output(broken)

    def test_rejects_lat_out_of_range(self, stub_ingestor: _StubIngestor) -> None:
        bad_lat = np.arange(30.0, 62.25, 0.25)  # Starts at 30°, not 6°
        bad_shape = (len(_LEAD_HOURS), len(bad_lat), len(_LON))
        coords = {"lead_hours": _LEAD_HOURS, "lat": bad_lat, "lon": _LON}
        data_vars = {
            v: xr.DataArray(
                np.ones(bad_shape, dtype=np.float32),
                dims=["lead_hours", "lat", "lon"],
                coords=coords,
            )
            for v in BaseIngestor.REQUIRED_VARS
        }
        broken = xr.Dataset(data_vars, attrs=_VALID_ATTRS)
        with pytest.raises(DataContractError, match="lat"):
            stub_ingestor.validate_output(broken)

    def test_rejects_lon_out_of_range(self, stub_ingestor: _StubIngestor) -> None:
        bad_lon = np.arange(0.0, 30.25, 0.25)   # Europe range, not India
        bad_shape = (len(_LEAD_HOURS), len(_LAT), len(bad_lon))
        coords = {"lead_hours": _LEAD_HOURS, "lat": _LAT, "lon": bad_lon}
        data_vars = {
            v: xr.DataArray(
                np.ones(bad_shape, dtype=np.float32),
                dims=["lead_hours", "lat", "lon"],
                coords=coords,
            )
            for v in BaseIngestor.REQUIRED_VARS
        }
        broken = xr.Dataset(data_vars, attrs=_VALID_ATTRS)
        with pytest.raises(DataContractError, match="lon"):
            stub_ingestor.validate_output(broken)

    def test_rejects_missing_model_name(
        self, stub_ingestor: _StubIngestor, valid_ds: xr.Dataset
    ) -> None:
        broken = valid_ds.copy()
        broken.attrs.pop("model_name")
        with pytest.raises(DataContractError, match="model_name"):
            stub_ingestor.validate_output(broken)

    def test_rejects_missing_units(
        self, stub_ingestor: _StubIngestor, valid_ds: xr.Dataset
    ) -> None:
        broken = valid_ds.copy()
        broken.attrs.pop("units")
        with pytest.raises(DataContractError, match="units"):
            stub_ingestor.validate_output(broken)

    def test_rejects_invalid_model_type(
        self, stub_ingestor: _StubIngestor, valid_ds: xr.Dataset
    ) -> None:
        broken = valid_ds.assign_attrs(model_type="bogus_type")
        with pytest.raises(DataContractError, match="model_type"):
            stub_ingestor.validate_output(broken)
