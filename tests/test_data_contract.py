"""
tests/test_data_contract.py
---------------------------
CHECK 1 — Data contract compliance.

Validates that _generate_synthetic_base() + _build_dataset() produce
xr.Dataset objects that satisfy every requirement of the Sacred Data
Contract, and that BaseIngestor.validate_output() correctly accepts
conformant datasets and rejects non-conformant ones.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from scripts.generate_demo_data import LEAD_HOURS, LAT, LON, SHAPE
from src.ingestion.base import BaseIngestor

REQUIRED_VARS: list[str] = ["t2m", "tp", "u10", "v10", "mslp"]


# ===========================================================================
# CHECK 1 — Positive path: conformant dataset must pass every assertion
# ===========================================================================

class TestDataContractCompliance:
    """Every Sacred Data Contract assertion must hold for a conformant dataset."""

    def test_isinstance_xr_dataset(self, sample_dataset: xr.Dataset) -> None:
        """Result must be an xr.Dataset, not a DataFrame or any other type."""
        assert isinstance(sample_dataset, xr.Dataset)

    def test_dimensions(self, sample_dataset: xr.Dataset) -> None:
        """Dataset must have exactly the three required dimensions."""
        assert set(sample_dataset.dims) == {"lead_hours", "lat", "lon"}

    def test_all_variables_present(self, sample_dataset: xr.Dataset) -> None:
        """All five contract variables must exist as data variables."""
        assert all(v in sample_dataset for v in REQUIRED_VARS)

    def test_model_name_attr_not_none(self, sample_dataset: xr.Dataset) -> None:
        """The model_name attribute must be present and non-None."""
        assert sample_dataset.attrs.get("model_name") is not None

    def test_model_type_attr_valid(self, sample_dataset: xr.Dataset) -> None:
        """model_type must be one of the three allowed values."""
        assert sample_dataset.attrs.get("model_type") in {"nwp", "ai", "ensemble"}

    def test_validate_output_returns_true(self, sample_dataset: xr.Dataset) -> None:
        """BaseIngestor.validate_output() must return True for a conformant dataset."""
        assert BaseIngestor.validate_output(sample_dataset) is True

    def test_lead_hours_exact_values(self, sample_dataset: xr.Dataset) -> None:
        """lead_hours coordinate must match [0, 24, 48, 72, 96, 120] exactly."""
        np.testing.assert_array_equal(
            sample_dataset.coords["lead_hours"].values,
            LEAD_HOURS,
        )

    def test_lat_coordinate_values(self, sample_dataset: xr.Dataset) -> None:
        """lat must run from 6.0 to 38.0 in 0.25° steps (129 points)."""
        assert len(sample_dataset.coords["lat"]) == len(LAT)
        np.testing.assert_allclose(
            sample_dataset.coords["lat"].values, LAT, atol=1e-6
        )

    def test_lon_coordinate_values(self, sample_dataset: xr.Dataset) -> None:
        """lon must run from 68.0 to 97.0 in 0.25° steps (117 points)."""
        assert len(sample_dataset.coords["lon"]) == len(LON)
        np.testing.assert_allclose(
            sample_dataset.coords["lon"].values, LON, atol=1e-6
        )

    def test_variable_shapes_match_contract(self, sample_dataset: xr.Dataset) -> None:
        """Every variable must have shape (n_leads=6, n_lat=129, n_lon=117)."""
        for var in REQUIRED_VARS:
            actual = sample_dataset[var].shape
            assert actual == SHAPE, (
                f"Variable '{var}': expected shape {SHAPE}, got {actual}."
            )

    def test_tp_non_negative(self, sample_dataset: xr.Dataset) -> None:
        """Precipitation (tp) must never be negative — physical constraint."""
        assert float(sample_dataset["tp"].min()) >= 0.0

    def test_t2m_plausible_range(self, sample_dataset: xr.Dataset) -> None:
        """t2m must be within a plausible India surface temperature range (K)."""
        assert float(sample_dataset["t2m"].min()) > 240.0, "t2m too cold"
        assert float(sample_dataset["t2m"].max()) < 340.0, "t2m too hot"

    def test_mslp_plausible_range(self, sample_dataset: xr.Dataset) -> None:
        """mslp must be near standard atmosphere (Pa)."""
        assert float(sample_dataset["mslp"].min()) > 95_000.0
        assert float(sample_dataset["mslp"].max()) < 108_000.0

    def test_units_attr_present(self, sample_dataset: xr.Dataset) -> None:
        """The 'units' attribute must be present (contract requires it)."""
        assert "units" in sample_dataset.attrs


# ===========================================================================
# CHECK 1 — Negative path: validate_output must reject broken datasets
# ===========================================================================

class TestValidateOutputRejectsInvalid:
    """BaseIngestor.validate_output() must raise for every contract violation."""

    def test_rejects_non_dataset_raises_type_error(self) -> None:
        """Passing a non-Dataset object must raise TypeError."""
        with pytest.raises(TypeError, match="xr.Dataset"):
            BaseIngestor.validate_output({"t2m": 1})  # type: ignore[arg-type]

    def test_rejects_missing_variable(self, sample_dataset: xr.Dataset) -> None:
        """A dataset missing t2m must be rejected with descriptive ValueError."""
        broken = sample_dataset.drop_vars("t2m")
        with pytest.raises(ValueError, match="Missing required variables"):
            BaseIngestor.validate_output(broken)

    def test_rejects_wrong_dimensions(self) -> None:
        """A dataset with wrong dimensions must be rejected."""
        bad = xr.Dataset(
            {"t2m": xr.DataArray(np.zeros((3, 3)), dims=["x", "y"])},
            attrs={"model_name": "bad", "model_type": "nwp"},
        )
        with pytest.raises(ValueError, match="Dimension mismatch"):
            BaseIngestor.validate_output(bad)

    def test_rejects_missing_model_name(self, sample_dataset: xr.Dataset) -> None:
        """A dataset without model_name attr must be rejected."""
        broken = sample_dataset.copy()
        broken.attrs.pop("model_name", None)
        with pytest.raises(ValueError, match="model_name"):
            BaseIngestor.validate_output(broken)

    def test_rejects_invalid_model_type(self, sample_dataset: xr.Dataset) -> None:
        """A dataset with an unrecognised model_type must be rejected."""
        broken = sample_dataset.assign_attrs(model_type="unknown_type")
        with pytest.raises(ValueError, match="model_type"):
            BaseIngestor.validate_output(broken)

    def test_rejects_wrong_lead_hours(self, sample_dataset: xr.Dataset) -> None:
        """A dataset whose lead_hours coord differs from contract must be rejected."""
        wrong_leads = [0, 6, 12, 18, 24, 30]  # 6-hourly instead of 24-hourly
        broken = sample_dataset.assign_coords(lead_hours=wrong_leads)
        with pytest.raises(ValueError, match="lead_hours"):
            BaseIngestor.validate_output(broken)
