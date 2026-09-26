"""
tests/unit/test_demo_ingestor.py
---------------------------------
Unit tests for DemoIngestor.ingest().

Uses tmp_path to write real NetCDF files so xr.open_dataset works
without any mocking.  IngestionError is asserted on a missing file.

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

from src.ingestion.base_ingestor import DataContractError, IngestionError  # noqa: E402
from src.ingestion.demo_ingestor import DemoIngestor  # noqa: E402

# ---------------------------------------------------------------------------
# Contract-compliant helpers (mirrors Sacred Data Contract exactly)
# ---------------------------------------------------------------------------

_LEAD_HOURS = [0, 24, 48, 72, 96, 120]
_LAT = np.arange(6.0, 38.25, 0.25)
_LON = np.arange(68.0, 97.25, 0.25)
_SHAPE = (len(_LEAD_HOURS), len(_LAT), len(_LON))
_COORDS = {"lead_hours": _LEAD_HOURS, "lat": _LAT, "lon": _LON}

_TEST_DATE = pd.Timestamp("2024-06-01")
_MODEL_KEY = "pangu"


def _build_valid_nc(path: Path) -> None:
    """Write a contract-compliant NetCDF file to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data_vars = {
        var: xr.DataArray(
            np.random.default_rng(42).random(_SHAPE).astype(np.float32),
            dims=["lead_hours", "lat", "lon"],
            coords=_COORDS,
        )
        for var in ("t2m", "tp", "u10", "v10", "mslp")
    }
    # tp must be non-negative
    data_vars["tp"] = xr.DataArray(
        np.abs(data_vars["tp"].values),
        dims=["lead_hours", "lat", "lon"],
        coords=_COORDS,
    )
    ds = xr.Dataset(
        data_vars=data_vars,
        attrs={
            "model_name": _MODEL_KEY,
            "model_type": "ai",
            "units": "{'t2m': 'K', 'tp': 'mm/hr', 'u10': 'm/s', 'v10': 'm/s', 'mslp': 'Pa'}",
        },
    )
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """Return a tmp directory that acts as the raw data root."""
    return tmp_path / "raw"


@pytest.fixture
def demo_config(raw_root: Path) -> dict:
    """Return a pipeline config pointing at tmp raw_root."""
    return {"paths": {"raw_data": str(raw_root)}}


@pytest.fixture
def demo_model_config() -> dict:
    """Return a model config dict for pangu with 'key' injected."""
    return {
        "name": "Pangu-Weather",
        "type": "ai",
        "source": "demo",
        "key": _MODEL_KEY,
    }


@pytest.fixture
def ingestor(demo_config: dict, demo_model_config: dict) -> DemoIngestor:
    """Return a DemoIngestor wired to the tmp raw_root."""
    return DemoIngestor(demo_config, demo_model_config)


@pytest.fixture
def nc_path(raw_root: Path) -> Path:
    """Write a valid NetCDF file and return its path."""
    path = raw_root / _MODEL_KEY / f"{_TEST_DATE.strftime('%Y%m%d')}.nc"
    _build_valid_nc(path)
    return path


# ===========================================================================
# DemoIngestor — successful ingest
# ===========================================================================

class TestDemoIngestorSuccess:
    def test_returns_xr_dataset(
        self, ingestor: DemoIngestor, nc_path: Path
    ) -> None:
        ds = ingestor.ingest(_TEST_DATE)
        assert isinstance(ds, xr.Dataset)

    def test_all_required_variables_present(
        self, ingestor: DemoIngestor, nc_path: Path
    ) -> None:
        ds = ingestor.ingest(_TEST_DATE)
        for var in ("t2m", "tp", "u10", "v10", "mslp"):
            assert var in ds, f"Variable '{var}' missing from result"

    def test_dimensions_are_correct(
        self, ingestor: DemoIngestor, nc_path: Path
    ) -> None:
        ds = ingestor.ingest(_TEST_DATE)
        assert set(ds.dims) == {"lead_hours", "lat", "lon"}

    def test_lead_hours_coord_values(
        self, ingestor: DemoIngestor, nc_path: Path
    ) -> None:
        ds = ingestor.ingest(_TEST_DATE)
        np.testing.assert_array_equal(ds.coords["lead_hours"].values, _LEAD_HOURS)

    def test_validate_output_passes(
        self, ingestor: DemoIngestor, nc_path: Path
    ) -> None:
        """validate_output must not raise for the returned dataset."""
        ds = ingestor.ingest(_TEST_DATE)
        assert ingestor.validate_output(ds) is True

    def test_logs_info_message(
        self,
        ingestor: DemoIngestor,
        nc_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Ingest must emit one INFO log with model key and date."""
        import logging
        with caplog.at_level(logging.INFO):
            ingestor.ingest(_TEST_DATE)
        assert any(
            _MODEL_KEY in r.message and "2024-06-01" in r.message
            for r in caplog.records
        )

    def test_uses_model_key_for_path(
        self, demo_config: dict, nc_path: Path
    ) -> None:
        """key field is used as directory name, not the full model name."""
        model_config = {
            "name": "Pangu-Weather",
            "type": "ai",
            "source": "demo",
            "key": _MODEL_KEY,
        }
        ingestor = DemoIngestor(demo_config, model_config)
        ds = ingestor.ingest(_TEST_DATE)
        assert isinstance(ds, xr.Dataset)


# ===========================================================================
# DemoIngestor — error handling
# ===========================================================================

class TestDemoIngestorErrors:
    def test_raises_ingestion_error_when_file_missing(
        self, ingestor: DemoIngestor
    ) -> None:
        """Missing file must raise IngestionError, not FileNotFoundError."""
        missing_date = pd.Timestamp("2099-01-01")
        with pytest.raises(IngestionError) as exc_info:
            ingestor.ingest(missing_date)
        assert "pangu" in str(exc_info.value)
        assert "2099-01-01" in str(exc_info.value) or "20990101" in str(exc_info.value)

    def test_ingestion_error_contains_path(self, ingestor: DemoIngestor) -> None:
        """IngestionError message must include the attempted file path."""
        with pytest.raises(IngestionError) as exc_info:
            ingestor.ingest(pd.Timestamp("2001-01-01"))
        assert str(exc_info.value).count("/") > 0 or str(exc_info.value).count("\\") > 0

    def test_raises_data_contract_error_for_broken_nc(
        self,
        ingestor: DemoIngestor,
        raw_root: Path,
    ) -> None:
        """A NetCDF that lacks required vars must raise DataContractError."""
        bad_path = raw_root / _MODEL_KEY / f"{_TEST_DATE.strftime('%Y%m%d')}.nc"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a file missing the 'tp' variable
        coords = {"lead_hours": _LEAD_HOURS, "lat": _LAT, "lon": _LON}
        ds = xr.Dataset(
            {
                "t2m": xr.DataArray(
                    np.zeros(_SHAPE, dtype=np.float32),
                    dims=["lead_hours", "lat", "lon"],
                    coords=coords,
                )
            },
            attrs={"model_name": _MODEL_KEY, "model_type": "ai", "units": "{}"},
        )
        ds.to_netcdf(bad_path)
        with pytest.raises(DataContractError):
            ingestor.ingest(_TEST_DATE)
