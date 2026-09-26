"""
tests/unit/test_variable_mapper.py
----------------------------------
Unit tests for VariableMapper.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import ast

import numpy as np
import xarray as xr

from src.preprocessing.variable_mapper import VariableMapper


def test_variable_mapper_rename() -> None:
    """Test that variables are renamed according to var_mapping."""
    mapper = VariableMapper()
    ds = xr.Dataset(
        {"TMP_2m": (["lat", "lon"], np.zeros((2, 2)))},
        attrs={"units": "{}"}
    )
    
    model_config = {
        "test": {
            "var_mapping": {"TMP_2m": "t2m"}
        }
    }
    
    out_ds = mapper.map(ds, "test", model_config)
    assert "t2m" in out_ds
    assert "TMP_2m" not in out_ds


def test_variable_mapper_temperature_celsius() -> None:
    """Test temperature conversion from Celsius to Kelvin."""
    mapper = VariableMapper()
    # 0 deg C should become 273.15 K
    ds = xr.Dataset(
        {"t2m": (["lat", "lon"], np.zeros((2, 2)))},
        attrs={"units": "{}"}
    )
    
    model_config = {
        "test": {
            "source_units": {"t2m": "C"}
        }
    }
    
    out_ds = mapper.map(ds, "test", model_config)
    assert np.allclose(out_ds["t2m"].values, 273.15)
    
    units = ast.literal_eval(out_ds.attrs["units"])
    assert units["t2m"] == "K"


def test_variable_mapper_precipitation_kgms() -> None:
    """Test precipitation conversion from kg/m2/s to mm/hr."""
    mapper = VariableMapper()
    # 1 kg/m2/s should become 3600 mm/hr
    ds = xr.Dataset(
        {"tp": (["lat", "lon"], np.ones((2, 2)))},
        attrs={"units": "{}"}
    )
    
    model_config = {
        "test": {
            "source_units": {"tp": "kg/m2/s"}
        }
    }
    
    out_ds = mapper.map(ds, "test", model_config)
    assert np.allclose(out_ds["tp"].values, 3600.0)
    
    units = ast.literal_eval(out_ds.attrs["units"])
    assert units["tp"] == "mm/hr"


def test_variable_mapper_no_conversion_needed() -> None:
    """Test that variables not requiring conversion are passed through."""
    mapper = VariableMapper()
    ds = xr.Dataset(
        {"u10": (["lat", "lon"], np.ones((2, 2)))},
        attrs={"units": "{}"}
    )
    
    model_config = {"test": {}}
    
    out_ds = mapper.map(ds, "test", model_config)
    assert np.allclose(out_ds["u10"].values, 1.0)
    
    units = ast.literal_eval(out_ds.attrs["units"])
    assert units["u10"] == "m/s"
