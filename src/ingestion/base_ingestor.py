"""
src/ingestion/base_ingestor.py
------------------------------
Layer   : ingestion
Purpose : Custom exceptions and abstract base class shared by every
          model ingestor in the Hybrid AI-NWP Blending System.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """Raised when data cannot be fetched or read from an upstream source."""


class DataContractError(Exception):
    """Raised when an xr.Dataset violates the Sacred Data Contract."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseIngestor(ABC):
    """Abstract base class for all model data ingestors.

    Subclasses implement :meth:`ingest` to fetch data from a specific source
    (NOMADS, ECMWF Open Data, demo files, etc.).  This class provides the
    authoritative :meth:`validate_output` implementation that every concrete
    ingestor must call before returning a dataset.

    Parameters
    ----------
    config : dict
        Full pipeline configuration loaded from ``config/pipeline_config.yaml``.
    model_config : dict
        Single model entry loaded from ``config/models.yaml``.
        Callers should inject the YAML key under ``model_config["key"]``
        (e.g. ``model_config["key"] = "pangu"``) before constructing.
    """

    # ------------------------------------------------------------------
    # Contract constants — single source of truth
    # ------------------------------------------------------------------
    REQUIRED_VARS: tuple[str, ...] = ("t2m", "tp", "u10", "v10", "mslp")
    REQUIRED_DIMS: frozenset[str] = frozenset({"lead_hours", "lat", "lon"})
    VALID_MODEL_TYPES: frozenset[str] = frozenset({"nwp", "ai", "ensemble"})
    EXPECTED_LEAD_HOURS: list[int] = [0, 24, 48, 72, 96, 120]
    REQUIRED_ATTRS: tuple[str, ...] = ("model_name", "model_type", "units")

    # Coordinate range centres (degrees) with allowed tolerance (±0.5°)
    _LAT_MIN: float = 6.0
    _LAT_MAX: float = 38.0
    _LON_MIN: float = 68.0
    _LON_MAX: float = 97.0
    _COORD_TOL: float = 0.5

    def __init__(self, config: dict, model_config: dict) -> None:
        """Store pipeline and model configuration.

        Parameters
        ----------
        config : dict
            Pipeline configuration (``paths``, ``scheduler``, ``api``).
        model_config : dict
            One model entry from ``models.yaml``.  Must contain at minimum
            ``name``, ``type``, and ``source`` keys.  Callers should also
            inject ``key`` (the short YAML identifier, e.g. ``"pangu"``).
        """
        self.config = config
        self.model_config = model_config

    @abstractmethod
    def ingest(self, date: pd.Timestamp, lead_hours: list[int]) -> xr.Dataset:
        """Fetch and return a contract-compliant dataset for *date*.

        Parameters
        ----------
        date : pd.Timestamp
            Forecast initialisation date.
        lead_hours : list[int]
            Lead hours to include in the returned dataset.

        Returns
        -------
        xr.Dataset
            Dataset conforming to the Sacred Data Contract.
        """

    def validate_output(self, ds: xr.Dataset) -> bool:
        """Validate *ds* against every requirement of the Sacred Data Contract.

        Checks are applied in sequence; the first failure raises immediately
        with a message that identifies the exact violation.  All-NaN slices
        emit a ``WARNING`` log but do *not* raise (data may be legitimately
        missing for some lead hours).

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to validate.

        Returns
        -------
        bool
            ``True`` when every check passes.

        Raises
        ------
        DataContractError
            On the first contract violation, with a descriptive message.
        """
        # --- 1. Type ---------------------------------------------------
        if not isinstance(ds, xr.Dataset):
            raise DataContractError(
                f"Expected xr.Dataset, got {type(ds).__name__}."
            )

        # --- 2. Variables ----------------------------------------------
        missing_vars = [v for v in self.REQUIRED_VARS if v not in ds]
        if missing_vars:
            raise DataContractError(
                f"Missing required variables: {missing_vars}. "
                f"Required: {list(self.REQUIRED_VARS)}."
            )

        # --- 3. Dimensions ---------------------------------------------
        actual_dims = set(ds.dims)
        if actual_dims != self.REQUIRED_DIMS:
            raise DataContractError(
                f"Dimension mismatch. "
                f"Expected {sorted(self.REQUIRED_DIMS)}, "
                f"got {sorted(actual_dims)}."
            )

        # --- 4. lead_hours coordinate values ---------------------------
        if not np.array_equal(
            ds.coords["lead_hours"].values, self.EXPECTED_LEAD_HOURS
        ):
            raise DataContractError(
                f"lead_hours coordinate mismatch. "
                f"Expected {self.EXPECTED_LEAD_HOURS}, "
                f"got {ds.coords['lead_hours'].values.tolist()}."
            )

        # --- 5. lat range (±0.5° tolerance) ----------------------------
        lat_min = float(ds.coords["lat"].min())
        lat_max = float(ds.coords["lat"].max())
        if not (
            self._LAT_MIN - self._COORD_TOL
            <= lat_min
            <= self._LAT_MIN + self._COORD_TOL
        ):
            raise DataContractError(
                f"lat min ({lat_min:.3f}°) outside expected "
                f"{self._LAT_MIN} ± {self._COORD_TOL}°."
            )
        if not (
            self._LAT_MAX - self._COORD_TOL
            <= lat_max
            <= self._LAT_MAX + self._COORD_TOL
        ):
            raise DataContractError(
                f"lat max ({lat_max:.3f}°) outside expected "
                f"{self._LAT_MAX} ± {self._COORD_TOL}°."
            )

        # --- 6. lon range (±0.5° tolerance) ----------------------------
        lon_min = float(ds.coords["lon"].min())
        lon_max = float(ds.coords["lon"].max())
        if not (
            self._LON_MIN - self._COORD_TOL
            <= lon_min
            <= self._LON_MIN + self._COORD_TOL
        ):
            raise DataContractError(
                f"lon min ({lon_min:.3f}°) outside expected "
                f"{self._LON_MIN} ± {self._COORD_TOL}°."
            )
        if not (
            self._LON_MAX - self._COORD_TOL
            <= lon_max
            <= self._LON_MAX + self._COORD_TOL
        ):
            raise DataContractError(
                f"lon max ({lon_max:.3f}°) outside expected "
                f"{self._LON_MAX} ± {self._COORD_TOL}°."
            )

        # --- 7. Required attributes ------------------------------------
        missing_attrs = [
            a for a in self.REQUIRED_ATTRS
            if ds.attrs.get(a) is None
        ]
        if missing_attrs:
            raise DataContractError(
                f"Missing required attributes: {missing_attrs}. "
                f"Required: {list(self.REQUIRED_ATTRS)}."
            )

        # --- 8. model_type value ---------------------------------------
        model_type = ds.attrs.get("model_type")
        if model_type not in self.VALID_MODEL_TYPES:
            raise DataContractError(
                f"Invalid model_type '{model_type}'. "
                f"Must be one of {sorted(self.VALID_MODEL_TYPES)}."
            )

        # --- 9. All-NaN slice check (warn only, never fail) ------------
        model_name = ds.attrs.get("model_name", "unknown")
        for var in self.REQUIRED_VARS:
            da = ds[var]
            for lead_hour in ds.coords["lead_hours"].values:
                slice_vals = da.sel(lead_hours=lead_hour).values
                if np.all(np.isnan(slice_vals)):
                    logger.warning(
                        "All-NaN slice detected: model='%s', var='%s', "
                        "lead_hours=%d. Data may be missing or corrupt.",
                        model_name,
                        var,
                        int(lead_hour),
                    )

        logger.debug(
            "validate_output passed: model_name='%s', model_type='%s'.",
            model_name,
            model_type,
        )
        return True
