"""
tests/unit/test_regridder.py
----------------------------
Unit tests for Regridder.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pytest
import xarray as xr

from src.preprocessing.regridder import Regridder


@pytest.fixture
def small_regridder() -> Regridder:
    """Regridder with a small 5x5 target grid."""
    target_lat = np.arange(10.0, 15.0, 1.0)
    target_lon = np.arange(70.0, 75.0, 1.0)
    return Regridder(target_lat=target_lat, target_lon=target_lon)


def test_regridder_linear_interp(small_regridder: Regridder) -> None:
    """Test regridding on a simple dataset."""
    src_lat = np.arange(9.0, 16.0, 1.0)
    src_lon = np.arange(69.0, 76.0, 1.0)
    
    data = np.ones((len(src_lat), len(src_lon)))
    ds = xr.Dataset(
        {"t2m": (["lat", "lon"], data)},
        coords={"lat": src_lat, "lon": src_lon},
        attrs={"model_name": "test"}
    )
    
    out_ds = small_regridder.regrid(ds)
    assert "t2m" in out_ds
    assert out_ds["t2m"].shape == (5, 5)
    assert np.allclose(out_ds.lat.values, np.arange(10.0, 15.0, 1.0))
    assert np.allclose(out_ds.lon.values, np.arange(70.0, 75.0, 1.0))
    assert np.allclose(out_ds["t2m"].values, 1.0)
    assert out_ds.attrs["model_name"] == "test"


def test_regridder_spacing_error(small_regridder: Regridder) -> None:
    """Test that source spacing > 1.0 raises ValueError."""
    src_lat = np.arange(10.0, 15.0, 1.5)  # spacing 1.5 > 1.0
    src_lon = np.arange(70.0, 75.0, 1.0)
    
    ds = xr.Dataset(
        {"t2m": (["lat", "lon"], np.zeros((len(src_lat), len(src_lon))))},
        coords={"lat": src_lat, "lon": src_lon}
    )
    
    with pytest.raises(ValueError, match="Source lat spacing"):
        small_regridder.regrid(ds)


def test_regridder_lead_hours(small_regridder: Regridder) -> None:
    """Test regridding with lead_hours dimension."""
    src_lat = np.arange(9.0, 16.0, 1.0)
    src_lon = np.arange(69.0, 76.0, 1.0)
    lead_hours = [0, 24]
    
    data = np.ones((2, len(src_lat), len(src_lon)))
    ds = xr.Dataset(
        {"t2m": (["lead_hours", "lat", "lon"], data)},
        coords={"lead_hours": lead_hours, "lat": src_lat, "lon": src_lon}
    )
    
    out_ds = small_regridder.regrid(ds)
    assert out_ds["t2m"].shape == (2, 5, 5)
    assert "lead_hours" in out_ds.coords


def test_regridder_descending_coords(small_regridder: Regridder) -> None:
    """Test regridding handles descending source coordinates."""
    src_lat = np.arange(16.0, 8.0, -1.0)  # descending
    src_lon = np.arange(69.0, 76.0, 1.0)  # ascending
    
    data = np.ones((len(src_lat), len(src_lon)))
    ds = xr.Dataset(
        {"t2m": (["lat", "lon"], data)},
        coords={"lat": src_lat, "lon": src_lon}
    )
    
    out_ds = small_regridder.regrid(ds)
    assert out_ds["t2m"].shape == (5, 5)
    assert np.all(np.isfinite(out_ds["t2m"].values))


def test_regridder_out_of_bounds_nan(small_regridder: Regridder) -> None:
    """Test that points outside source domain become NaN."""
    # Source only covers lat 10.0 to 11.5
    src_lat = np.arange(10.0, 12.0, 0.5) 
    src_lon = np.arange(70.0, 75.0, 0.5)
    
    data = np.ones((len(src_lat), len(src_lon)))
    ds = xr.Dataset(
        {"t2m": (["lat", "lon"], data)},
        coords={"lat": src_lat, "lon": src_lon}
    )
    
    out_ds = small_regridder.regrid(ds)
    # Target lat goes up to 14.0. Points at lat >= 12.0 should be NaN.
    assert np.isnan(out_ds["t2m"].sel(lat=13.0).values).all()
    assert not np.isnan(out_ds["t2m"].sel(lat=11.0).values).any()
