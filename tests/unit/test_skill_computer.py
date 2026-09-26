"""
tests/unit/test_skill_computer.py
---------------------------------
Unit tests for the SkillComputer orchestrator.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.skill.skill_computer import SkillComputer


def test_skill_computer_compute() -> None:
    """Test full computation of metrics over models, regions, and lead hours."""
    # Setup configuration with one specific region
    config = {}
    regions = {
        "regions": {
            "test_region": {
                "lat_min": 10.0,
                "lat_max": 20.0,
                "lon_min": 70.0,
                "lon_max": 80.0
            }
        }
    }
    
    computer = SkillComputer(config, regions)
    
    # Check region masks were created
    assert "test_region" in computer.region_masks
    assert "all_india" in computer.region_masks
    
    # Contract standard coords
    target_lat = np.arange(6.0, 38.25, 0.25)
    target_lon = np.arange(68.0, 97.25, 0.25)
    
    # 2 lead hours
    shape = (2, len(target_lat), len(target_lon))
    
    # Truth dataset
    t_ds = xr.Dataset(
        {"t2m": (["lead_hours", "lat", "lon"], np.ones(shape))},
        coords={
            "lead_hours": [0, 24],
            "lat": target_lat,
            "lon": target_lon,
            "valid_time": pd.Timestamp("2024-01-15")  # January -> Season DJF
        }
    )
    
    # Forecast dataset (constant error of +2.0)
    f_ds = xr.Dataset(
        {"t2m": (["lead_hours", "lat", "lon"], np.full(shape, 3.0))},
        coords={
            "lead_hours": [0, 24],
            "lat": target_lat,
            "lon": target_lon,
            "valid_time": pd.Timestamp("2024-01-15")
        }
    )
    
    forecast_archive = {"ModelA": [f_ds]}
    truth_archive = [t_ds]
    
    # Run compute
    df = computer.compute(forecast_archive, truth_archive)
    
    # Assertions
    assert isinstance(df, pd.DataFrame)
    
    # 2 lead hours x 2 regions (test_region, all_india) -> 4 rows
    assert len(df) == 4 
    
    assert all(df["model"] == "ModelA")
    assert all(df["variable"] == "t2m")
    assert all(df["season"] == "DJF")
    
    # Error metrics should all exactly equal 2.0
    assert np.allclose(df["rmse"].values, 2.0)
    assert np.allclose(df["mae"].values, 2.0)
    assert np.allclose(df["bias"].values, 2.0)


def test_skill_computer_season_mapping() -> None:
    """Test that months map correctly to DJF, MAM, JJA, SON."""
    computer = SkillComputer({}, {})
    
    assert computer._get_season(1) == "DJF"
    assert computer._get_season(2) == "DJF"
    assert computer._get_season(12) == "DJF"
    assert computer._get_season(4) == "MAM"
    assert computer._get_season(7) == "JJA"
    assert computer._get_season(11) == "SON"
    assert computer._get_season(99) == "Unknown"
