"""
tests/unit/test_quality_control.py
----------------------------------
Unit tests for QualityControl.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import ast

import numpy as np
import pytest
import xarray as xr

from src.preprocessing.quality_control import QualityControl


def test_quality_control_clean_data() -> None:
    """Test that clean data is not flagged."""
    qc = QualityControl()
    
    ds = xr.Dataset(
        {"t2m": (["lat", "lon"], np.full((5, 5), 300.0))},
    )
    
    out_ds = qc.check(ds)
    assert out_ds.attrs["qc_applied"] is True
    
    counts = ast.literal_eval(out_ds.attrs["qc_flagged_counts"])
    assert "t2m" not in counts
    assert not np.isnan(out_ds["t2m"].values).any()


def test_quality_control_flags_outliers() -> None:
    """Test that individual outliers are set to NaN."""
    qc = QualityControl()
    
    data = np.full((5, 5), 300.0)
    data[0, 0] = 400.0  # Above 340.0 K
    
    ds = xr.Dataset(
        {"t2m": (["lat", "lon"], data)},
    )
    
    out_ds = qc.check(ds)
    assert np.isnan(out_ds["t2m"].values[0, 0])
    assert not np.isnan(out_ds["t2m"].values[1, 1])
    
    counts = ast.literal_eval(out_ds.attrs["qc_flagged_counts"])
    assert counts["t2m"] == 1


def test_quality_control_warning_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """Test that warning is logged when > 5% of points are flagged."""
    qc = QualityControl()
    
    # 5x5 = 25 pts. 5% is 1.25 pts. 2 outliers = 8% -> WARNING
    data = np.full((5, 5), 300.0)
    data[0, 0] = 400.0
    data[0, 1] = 100.0
    
    ds = xr.Dataset(
        {"t2m": (["lat", "lon"], data)},
    )
    
    qc.check(ds)
    assert any("QC Warning: t2m has 8.0% outliers." in r.message for r in caplog.records)


def test_quality_control_error_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """Test that error is logged when > 30% of points are flagged."""
    qc = QualityControl()
    
    # 5x5 = 25 pts. 30% is 7.5 pts. 8 outliers = 32% -> ERROR
    data = np.full((5, 5), 300.0)
    data[0:2, 0:4] = 400.0  # 8 outliers
    
    ds = xr.Dataset(
        {"t2m": (["lat", "lon"], data)},
    )
    
    qc.check(ds)
    assert any("QC Error: t2m has 32.0% outliers." in r.message for r in caplog.records)


def test_quality_control_lead_hours(caplog: pytest.LogCaptureFixture) -> None:
    """Test that QC handles lead_hours dimension correctly."""
    qc = QualityControl()
    
    # Shape: (2, 5, 5) -> 2 lead hours, 25 pts each
    data = np.full((2, 5, 5), 300.0)
    # Lead hour 0: 2 outliers -> 8% -> Warning
    data[0, 0, 0] = 400.0
    data[0, 0, 1] = 400.0
    # Lead hour 24: 8 outliers -> 32% -> Error
    data[1, 0:2, 0:4] = 400.0
    
    ds = xr.Dataset(
        {"t2m": (["lead_hours", "lat", "lon"], data)},
        coords={"lead_hours": [0, 24]}
    )
    
    qc.check(ds)
    
    assert any("QC Warning: t2m at lead_hour 0 has 8.0% outliers" in r.message for r in caplog.records)
    assert any("QC Error: t2m at lead_hour 24 has 32.0% outliers" in r.message for r in caplog.records)
