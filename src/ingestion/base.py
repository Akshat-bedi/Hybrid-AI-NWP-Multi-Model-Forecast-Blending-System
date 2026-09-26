"""
src/ingestion/base.py
---------------------
Layer   : ingestion
Purpose : Abstract base class shared by all model ingestors.
          Provides the authoritative Sacred Data Contract validator used
          throughout the test suite and concrete ingestor subclasses.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sacred Data Contract constants (single source of truth for validation)
# ---------------------------------------------------------------------------
_REQUIRED_VARS: tuple[str, ...] = ("t2m", "tp", "u10", "v10", "mslp")
_REQUIRED_DIMS: frozenset[str] = frozenset({"lead_hours", "lat", "lon"})
_VALID_MODEL_TYPES: frozenset[str] = frozenset({"nwp", "ai", "ensemble"})
_LEAD_HOURS: list[int] = [0, 24, 48, 72, 96, 120]
_LAT: np.ndarray = np.arange(6.0, 38.25, 0.25)   # 129 points
_LON: np.ndarray = np.arange(68.0, 97.25, 0.25)  # 117 points


class BaseIngestor(ABC):
    """Abstract base class for all model data ingestors.

    Concrete subclasses implement :meth:`fetch` and :meth:`parse`.
    This class supplies the shared :meth:`validate_output` contract checker
    that every ingestor result must satisfy.
    """

    @staticmethod
    def validate_output(ds: xr.Dataset) -> bool:
        """Validate *ds* against the Sacred Data Contract.

        Checks are applied in order; the first failure raises immediately
        with a descriptive message identifying the exact violation.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to validate.

        Returns
        -------
        bool
            ``True`` when every contract requirement is satisfied.

        Raises
        ------
        TypeError
            If *ds* is not an ``xr.Dataset``.
        ValueError
            On the first contract violation, with a message describing it.
        """
        # --- Type -------------------------------------------------------
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"Expected xr.Dataset, got {type(ds).__name__}."
            )

        # --- Dimensions -------------------------------------------------
        actual_dims = set(ds.dims)
        if actual_dims != _REQUIRED_DIMS:
            raise ValueError(
                f"Dimension mismatch. "
                f"Expected {sorted(_REQUIRED_DIMS)}, got {sorted(actual_dims)}."
            )

        # --- Variables --------------------------------------------------
        missing = [v for v in _REQUIRED_VARS if v not in ds]
        if missing:
            raise ValueError(f"Missing required variables: {missing}.")

        # --- Coordinates ------------------------------------------------
        if not np.array_equal(ds.coords["lead_hours"].values, _LEAD_HOURS):
            raise ValueError(
                f"lead_hours coordinate mismatch. "
                f"Expected {_LEAD_HOURS}, "
                f"got {ds.coords['lead_hours'].values.tolist()}."
            )
        if not np.allclose(ds.coords["lat"].values, _LAT, atol=1e-6):
            raise ValueError(
                "lat coordinate does not match contract "
                "(expected arange(6.0, 38.25, 0.25), 129 points)."
            )
        if not np.allclose(ds.coords["lon"].values, _LON, atol=1e-6):
            raise ValueError(
                "lon coordinate does not match contract "
                "(expected arange(68.0, 97.25, 0.25), 117 points)."
            )

        # --- Attributes -------------------------------------------------
        if ds.attrs.get("model_name") is None:
            raise ValueError("Missing required attribute 'model_name'.")

        model_type = ds.attrs.get("model_type")
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. "
                f"Must be one of {sorted(_VALID_MODEL_TYPES)}."
            )

        logger.debug(
            "validate_output passed for model_name='%s', model_type='%s'.",
            ds.attrs["model_name"],
            model_type,
        )
        return True

    @abstractmethod
    def fetch(self, **kwargs: Any) -> None:
        """Fetch raw data from the upstream source."""

    @abstractmethod
    def parse(self, **kwargs: Any) -> xr.Dataset:
        """Parse fetched raw data into a contract-compliant ``xr.Dataset``."""
