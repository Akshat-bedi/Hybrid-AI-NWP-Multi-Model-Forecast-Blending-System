"""
src/ingestion/demo_ingestor.py
------------------------------
Layer   : ingestion
Purpose : Reads pre-generated NetCDF demo files (written by
          scripts/generate_demo_data.py) for AI / demo-source models
          such as Pangu-Weather and GraphCast.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from src.ingestion.base_ingestor import BaseIngestor, IngestionError

logger = logging.getLogger(__name__)


class DemoIngestor(BaseIngestor):
    """Ingestor for pre-generated demo NetCDF files.

    Expected file layout on disk::

        {config["paths"]["raw_data"]}/{model_key}/{YYYYMMDD}.nc

    where ``model_key`` is the short YAML identifier (e.g. ``"pangu"``).
    Callers must inject ``model_config["key"]`` before constructing this
    ingestor (see :class:`~src.ingestion.base_ingestor.BaseIngestor`).

    Raises
    ------
    IngestionError
        If the NetCDF file for the requested date does not exist.
    DataContractError
        If the loaded dataset fails :meth:`validate_output`.
    """

    def ingest(
        self,
        date: pd.Timestamp,
        lead_hours: list[int] | None = None,
    ) -> xr.Dataset:
        """Load a demo model NetCDF and return a validated dataset.

        Parameters
        ----------
        date : pd.Timestamp
            Forecast initialisation date used to locate the file.
        lead_hours : list[int] | None, optional
            Lead hours to expose in the dataset.  Defaults to
            ``[0, 24, 48, 72, 96, 120]`` when *None*.

        Returns
        -------
        xr.Dataset
            Contract-compliant dataset for *date*.

        Raises
        ------
        IngestionError
            When the expected NetCDF file is absent from the data store.
        DataContractError
            When the loaded file violates the Sacred Data Contract.
        """
        if lead_hours is None:
            lead_hours = [0, 24, 48, 72, 96, 120]

        # Derive the short YAML key (e.g. "pangu") from model_config.
        # Callers should inject model_config["key"]; fall back to "name".
        model_key: str = self.model_config.get(
            "key", self.model_config["name"]
        )

        raw_root = Path(self.config["paths"]["raw_data"])
        nc_path = raw_root / model_key / f"{date.strftime('%Y%m%d')}.nc"

        if not nc_path.exists():
            raise IngestionError(
                f"Demo NetCDF file not found for model '{model_key}' "
                f"on {date.strftime('%Y-%m-%d')}: {nc_path}"
            )

        ds: xr.Dataset = xr.open_dataset(nc_path)
        logger.info("Loaded %s for %s", model_key, f"{date:%Y-%m-%d}")

        self.validate_output(ds)
        return ds
