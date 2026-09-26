"""
tests/unit/test_gfs_ingestor.py
--------------------------------
Unit tests for GFSIngestor.

All network and cfgrib calls are mocked — no real HTTP traffic or
GRIB2 files are needed:

  - ``urllib.request.urlopen`` → returns a fake response with dummy bytes
  - ``cfgrib.open_datasets``    → injected via ``sys.modules`` (lazy import)

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.base_ingestor import DataContractError, IngestionError  # noqa: E402
from src.ingestion.gfs_ingestor import GFSIngestor  # noqa: E402

# ---------------------------------------------------------------------------
# Shared contract-sized arrays for mock cfgrib output
# ---------------------------------------------------------------------------

_LAT_DESC = np.arange(38.0, 5.75, -0.25)   # Descending (GRIB convention)
_LON = np.arange(68.0, 97.25, 0.25)
_LEAD_HOURS = [0, 24, 48, 72, 96, 120]
_DATE = pd.Timestamp("2024-06-01")


def _fake_cfgrib_datasets() -> list[xr.Dataset]:
    """Return a list of xr.Datasets that mimic cfgrib.open_datasets output.

    Each dataset contains a subset of GFS_VAR_MAP variables with realistic
    descending-latitude coordinates (as GRIB2 files typically carry).
    """
    shape = (len(_LAT_DESC), len(_LON))
    coords = {"latitude": _LAT_DESC, "longitude": _LON}

    ds_t = xr.Dataset(
        {
            "TMP_2maboveground": xr.DataArray(
                np.full(shape, 300.0, dtype=np.float32),
                dims=["latitude", "longitude"],
                coords=coords,
            )
        }
    )
    ds_wind = xr.Dataset(
        {
            "UGRD_10maboveground": xr.DataArray(
                np.full(shape, 3.0, dtype=np.float32),
                dims=["latitude", "longitude"],
                coords=coords,
            ),
            "VGRD_10maboveground": xr.DataArray(
                np.full(shape, 2.0, dtype=np.float32),
                dims=["latitude", "longitude"],
                coords=coords,
            ),
        }
    )
    ds_precip = xr.Dataset(
        {
            "PRATE_surface": xr.DataArray(
                np.full(shape, 1e-4, dtype=np.float32),  # kg/m²/s
                dims=["latitude", "longitude"],
                coords=coords,
            )
        }
    )
    ds_mslp = xr.Dataset(
        {
            "PRMSL_meansealevel": xr.DataArray(
                np.full(shape, 101_325.0, dtype=np.float32),
                dims=["latitude", "longitude"],
                coords=coords,
            )
        }
    )
    return [ds_t, ds_wind, ds_precip, ds_mslp]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """Temporary raw data directory."""
    return tmp_path / "raw"


@pytest.fixture
def gfs_config(raw_root: Path) -> dict:
    """Pipeline config pointing at tmp raw_root."""
    return {"paths": {"raw_data": str(raw_root)}}


@pytest.fixture
def gfs_model_config() -> dict:
    """Minimal GFS model config with key injected."""
    return {
        "name": "Global Forecast System",
        "type": "nwp",
        "source": "nomads",
        "key": "gfs",
    }


@pytest.fixture
def ingestor(gfs_config: dict, gfs_model_config: dict) -> GFSIngestor:
    """GFSIngestor wired to tmp raw_root."""
    return GFSIngestor(gfs_config, gfs_model_config)


@pytest.fixture
def fake_urlopen_response() -> MagicMock:
    """Context-manager mock that returns 16 dummy bytes on .read()."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"\x00" * 16
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_response)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm


@pytest.fixture
def mock_cfgrib_module() -> MagicMock:
    """A MagicMock that impersonates the cfgrib module."""
    mock = MagicMock()
    mock.open_datasets.return_value = _fake_cfgrib_datasets()
    return mock


# ===========================================================================
# _build_url
# ===========================================================================

class TestBuildUrl:
    def test_url_contains_date(self, ingestor: GFSIngestor) -> None:
        url = ingestor._build_url(_DATE, run_hour=0, lead_hour=24)
        assert "20240601" in url

    def test_url_contains_run_hour_zero_padded(
        self, ingestor: GFSIngestor
    ) -> None:
        url = ingestor._build_url(_DATE, run_hour=6, lead_hour=0)
        assert "/06/" in url
        assert "t06z" in url

    def test_url_contains_lead_hour_zero_padded(
        self, ingestor: GFSIngestor
    ) -> None:
        url = ingestor._build_url(_DATE, run_hour=0, lead_hour=48)
        assert ".f048" in url

    def test_url_starts_with_nomads_base(self, ingestor: GFSIngestor) -> None:
        url = ingestor._build_url(_DATE, run_hour=0, lead_hour=0)
        assert url.startswith("https://nomads.ncep.noaa.gov")

    def test_url_three_digit_lead_hour(self, ingestor: GFSIngestor) -> None:
        url = ingestor._build_url(_DATE, run_hour=0, lead_hour=120)
        assert ".f120" in url

    def test_url_correct_format(self, ingestor: GFSIngestor) -> None:
        url = ingestor._build_url(_DATE, run_hour=0, lead_hour=24)
        expected_suffix = "gfs.20240601/00/atmos/gfs.t00z.pgrb2.0p25.f024"
        assert url.endswith(expected_suffix)


# ===========================================================================
# _download_grib
# ===========================================================================

class TestDownloadGrib:
    def test_downloads_on_first_attempt(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
        fake_urlopen_response: MagicMock,
    ) -> None:
        dest = tmp_path / "test.grib2"
        with patch("urllib.request.urlopen", return_value=fake_urlopen_response):
            result = ingestor._download_grib("http://fake.url/test.grib2", dest)
        assert result == dest
        assert dest.exists()

    def test_file_content_matches_response(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
        fake_urlopen_response: MagicMock,
    ) -> None:
        dest = tmp_path / "test.grib2"
        with patch("urllib.request.urlopen", return_value=fake_urlopen_response):
            ingestor._download_grib("http://fake.url/test.grib2", dest)
        assert dest.read_bytes() == b"\x00" * 16

    def test_retries_on_failure_then_succeeds(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
        fake_urlopen_response: MagicMock,
    ) -> None:
        """Should succeed on third attempt after two URLError failures."""
        import urllib.error

        dest = tmp_path / "retry.grib2"
        side_effects: list[Any] = [
            urllib.error.URLError("connection refused"),
            urllib.error.URLError("timeout"),
            fake_urlopen_response,
        ]
        with patch("urllib.request.urlopen", side_effect=side_effects), \
             patch("time.sleep"):  # Skip real sleep
            result = ingestor._download_grib("http://fake.url/retry.grib2", dest)
        assert result == dest

    def test_raises_ingestion_error_after_all_retries(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
    ) -> None:
        """IngestionError must be raised when all 3 attempts fail."""
        import urllib.error

        dest = tmp_path / "fail.grib2"
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("always fails"),
        ), patch("time.sleep"):
            with pytest.raises(IngestionError, match="3 attempts"):
                ingestor._download_grib("http://bad.url/fail.grib2", dest)

    def test_sleep_called_between_retries(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
    ) -> None:
        """time.sleep must be called between (not after) retry attempts."""
        import urllib.error

        dest = tmp_path / "sleep_test.grib2"
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("fail"),
        ), patch("time.sleep") as mock_sleep:
            with pytest.raises(IngestionError):
                ingestor._download_grib("http://bad.url", dest)
        # sleep called _MAX_RETRIES - 1 = 2 times (not after last attempt)
        assert mock_sleep.call_count == 2


# ===========================================================================
# _parse_grib
# ===========================================================================

class TestParseGrib:
    def _call_parse(
        self, ingestor: GFSIngestor, grib_path: Path, mock_cfgrib: MagicMock
    ) -> xr.Dataset:
        """Call _parse_grib with cfgrib injected via sys.modules."""
        with patch.dict(sys.modules, {"cfgrib": mock_cfgrib}):
            return ingestor._parse_grib(grib_path)

    def test_returns_xr_dataset(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
        mock_cfgrib_module: MagicMock,
    ) -> None:
        grib_path = tmp_path / "test.grib2"
        grib_path.write_bytes(b"\x00")
        ds = self._call_parse(ingestor, grib_path, mock_cfgrib_module)
        assert isinstance(ds, xr.Dataset)

    def test_all_contract_variables_present(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
        mock_cfgrib_module: MagicMock,
    ) -> None:
        grib_path = tmp_path / "test.grib2"
        grib_path.write_bytes(b"\x00")
        ds = self._call_parse(ingestor, grib_path, mock_cfgrib_module)
        for var in ("t2m", "tp", "u10", "v10", "mslp"):
            assert var in ds, f"'{var}' missing after _parse_grib"

    def test_coordinates_renamed_lat_lon(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
        mock_cfgrib_module: MagicMock,
    ) -> None:
        grib_path = tmp_path / "test.grib2"
        grib_path.write_bytes(b"\x00")
        ds = self._call_parse(ingestor, grib_path, mock_cfgrib_module)
        assert "lat" in ds.coords, "'latitude' was not renamed to 'lat'"
        assert "lon" in ds.coords, "'longitude' was not renamed to 'lon'"
        assert "latitude" not in ds.coords
        assert "longitude" not in ds.coords

    def test_cropped_to_india_lat_range(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
        mock_cfgrib_module: MagicMock,
    ) -> None:
        grib_path = tmp_path / "test.grib2"
        grib_path.write_bytes(b"\x00")
        ds = self._call_parse(ingestor, grib_path, mock_cfgrib_module)
        lat_vals = ds.coords["lat"].values
        assert float(lat_vals.min()) >= 6.0 - 0.5
        assert float(lat_vals.max()) <= 38.0 + 0.5

    def test_prate_converted_to_mm_per_hr(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
        mock_cfgrib_module: MagicMock,
    ) -> None:
        """PRATE 1e-4 kg/m²/s → 0.36 mm/hr (× 3600)."""
        grib_path = tmp_path / "test.grib2"
        grib_path.write_bytes(b"\x00")
        ds = self._call_parse(ingestor, grib_path, mock_cfgrib_module)
        expected_tp = 1e-4 * 3600.0
        np.testing.assert_allclose(
            ds["tp"].values, expected_tp, rtol=1e-5,
            err_msg="PRATE unit conversion failed (expected × 3600)"
        )

    def test_raises_ingestion_error_when_no_vars_found(
        self,
        ingestor: GFSIngestor,
        tmp_path: Path,
    ) -> None:
        """If cfgrib returns no recognised variables, raise IngestionError."""
        empty_cfgrib = MagicMock()
        empty_cfgrib.open_datasets.return_value = [
            xr.Dataset({"unrelated_var": xr.DataArray([1.0])})
        ]
        grib_path = tmp_path / "empty.grib2"
        grib_path.write_bytes(b"\x00")
        with patch.dict(sys.modules, {"cfgrib": empty_cfgrib}):
            with pytest.raises(IngestionError, match="No recognizable GFS variables"):
                ingestor._parse_grib(grib_path)


# ===========================================================================
# ingest (end-to-end with mocked download + parse)
# ===========================================================================

class TestGFSIngest:
    def _run_ingest(
        self,
        ingestor: GFSIngestor,
        mock_cfgrib: MagicMock,
        lead_hours: list[int] | None = None,
    ) -> xr.Dataset:
        """Run ingest() with all I/O mocked."""
        fake_cm = MagicMock()
        fake_cm.__enter__ = MagicMock(return_value=MagicMock(read=lambda: b"\x00"))
        fake_cm.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=fake_cm), \
             patch.dict(sys.modules, {"cfgrib": mock_cfgrib}):
            return ingestor.ingest(
                _DATE,
                lead_hours=lead_hours or [0, 24, 48, 72, 96, 120],
            )

    def test_returns_xr_dataset(
        self, ingestor: GFSIngestor, mock_cfgrib_module: MagicMock
    ) -> None:
        ds = self._run_ingest(ingestor, mock_cfgrib_module)
        assert isinstance(ds, xr.Dataset)

    def test_lead_hours_dimension_correct(
        self, ingestor: GFSIngestor, mock_cfgrib_module: MagicMock
    ) -> None:
        ds = self._run_ingest(ingestor, mock_cfgrib_module)
        np.testing.assert_array_equal(ds.coords["lead_hours"].values, _LEAD_HOURS)

    def test_model_attrs_set(
        self, ingestor: GFSIngestor, mock_cfgrib_module: MagicMock
    ) -> None:
        ds = self._run_ingest(ingestor, mock_cfgrib_module)
        assert ds.attrs.get("model_name") == "gfs"
        assert ds.attrs.get("model_type") == "nwp"
        assert ds.attrs.get("units") is not None

    def test_validate_output_passes(
        self, ingestor: GFSIngestor, mock_cfgrib_module: MagicMock
    ) -> None:
        ds = self._run_ingest(ingestor, mock_cfgrib_module)
        assert ingestor.validate_output(ds) is True

    def test_download_called_once_per_lead_hour(
        self, ingestor: GFSIngestor, mock_cfgrib_module: MagicMock
    ) -> None:
        """urlopen must be called exactly once for each requested lead hour.

        We pass the full contract lead_hours so validate_output does not
        reject the merged dataset for a mismatched coordinate.
        """
        fake_cm = MagicMock()
        fake_cm.__enter__ = MagicMock(return_value=MagicMock(read=lambda: b"\x00"))
        fake_cm.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=fake_cm) as mock_urlopen, \
             patch.dict(sys.modules, {"cfgrib": mock_cfgrib_module}):
            ingestor.ingest(_DATE, lead_hours=[0, 24, 48, 72, 96, 120])

        assert mock_urlopen.call_count == 6

    def test_grib_dir_created_under_raw_root(
        self,
        ingestor: GFSIngestor,
        mock_cfgrib_module: MagicMock,
        raw_root: Path,
    ) -> None:
        """GRIB files must be stored under raw_data/gfs/{YYYYMMDD}/.

        We call ingest() with the full lead_hours list so validate_output
        passes, then verify the GRIB directory was created.
        """
        fake_cm = MagicMock()
        fake_cm.__enter__ = MagicMock(return_value=MagicMock(read=lambda: b"\x00"))
        fake_cm.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=fake_cm), \
             patch.dict(sys.modules, {"cfgrib": mock_cfgrib_module}):
            ingestor.ingest(_DATE, lead_hours=[0, 24, 48, 72, 96, 120])

        expected_dir = raw_root / "gfs" / "20240601"
        assert expected_dir.exists()
