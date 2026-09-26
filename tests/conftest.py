"""
tests/conftest.py
-----------------
Shared pytest fixtures for the Hybrid AI-NWP Blending System test suite.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest
import xarray as xr

# Ensure project root is on sys.path regardless of invocation directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_demo_data import (  # noqa: E402
    _build_dataset,
    _generate_synthetic_base,
)
from src.utils.config_loader import load_config  # noqa: E402


@pytest.fixture(scope="session")
def sample_dataset() -> xr.Dataset:
    """Minimal contract-compliant ``xr.Dataset`` built in-memory.

    Uses a fixed reference date (2024-06-01) so the fixture is fully
    deterministic across the test session.
    """
    arrays = _generate_synthetic_base(date(2024, 6, 1))
    return _build_dataset(arrays, model_name="test_model", model_type="nwp")


@pytest.fixture(scope="session")
def blend_config() -> dict:
    """Parsed ``config/blend_config.yaml`` as a plain dict."""
    return load_config(str(PROJECT_ROOT / "config" / "blend_config.yaml"))


@pytest.fixture(scope="session")
def pipeline_config() -> dict:
    """Parsed ``config/pipeline_config.yaml`` as a plain dict."""
    return load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
