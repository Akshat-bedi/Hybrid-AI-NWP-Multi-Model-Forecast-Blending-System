"""
src/ingestion/gfs_ingestor.py
------------------------------
Layer   : ingestion
Purpose : Fetches GFS 0.25° GRIB2 forecasts from NOMADS, parses them with
          cfgrib, crops to the India domain, and returns contract-compliant
          xr.Datasets.

Note on HTTP downloads
----------------------
Downloads use the stdlib ``urllib.request`` module (no third-party HTTP
library required) with up to 3 retry attempts and a 5-second back-off.

Note on cfgrib
--------------
``cfgrib`` is imported lazily inside ``_parse_grib`` because it requires
the ``eccodes`` system library.  Tests mock it via ``sys.modules``.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.ingestion.base_ingestor import BaseIngestor, IngestionError

logger = logging.getLogger(__name__)

# NOMADS base URL (no trailing slash)
_NOMADS_BASE = (
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
)

# India domain bounding box (matches Sacred Data Contract)
_LAT_SOUTH: float = 6.0
_LAT_NORTH: float = 38.0
_LON_WEST: float = 68.0
_LON_EAST: float = 97.0

# Download retry parameters
_MAX_RETRIES: int = 3
_RETRY_SLEEP_S: int = 5


class GFSIngestor(BaseIngestor):
    """Ingestor for GFS 0.25° GRIB2 data from NOMADS.

    Downloads one GRIB2 file per lead hour, parses it with ``cfgrib``,
    renames variables to contract names, crops to India, and merges all
    lead hours into a single :class:`xr.Dataset`.

    Parameters
    ----------
    config : dict
        Full pipeline configuration (``config/pipeline_config.yaml``).
    model_config : dict
        GFS model entry from ``config/models.yaml`` with ``key="gfs"``
        injected by the caller.
    """

    # ------------------------------------------------------------------
    # GFS → contract variable name mapping
    # These are the names cfgrib assigns when reading GFS pgrb2 files.
    # ------------------------------------------------------------------
    GFS_VAR_MAP: dict[str, str] = {
        "TMP_2maboveground":   "t2m",
        "UGRD_10maboveground": "u10",
        "VGRD_10maboveground": "v10",
        "PRATE_surface":       "tp",
        "PRMSL_meansealevel":  "mslp",
    }

    # ------------------------------------------------------------------
    # URL builder
    # ------------------------------------------------------------------

    def _build_url(self, date: pd.Timestamp, run_hour: int, lead_hour: int) -> str:
        """Construct the NOMADS download URL for a single GFS file.

        Parameters
        ----------
        date : pd.Timestamp
            Forecast initialisation date.
        run_hour : int
            Model run hour (0, 6, 12, or 18).
        lead_hour : int
            Forecast lead hour (0 – 384).

        Returns
        -------
        str
            Full NOMADS URL for the requested GRIB2 file.
        """
        date_str = date.strftime("%Y%m%d")
        filename = (
            f"gfs.t{run_hour:02d}z.pgrb2.0p25.f{lead_hour:03d}"
        )
        url = (
            f"{_NOMADS_BASE}/gfs.{date_str}/{run_hour:02d}/atmos/{filename}"
        )
        logger.debug("Built GFS URL: %s", url)
        return url

    # ------------------------------------------------------------------
    # GRIB download
    # ------------------------------------------------------------------

    def _download_grib(self, url: str, local_path: Path) -> Path:
        """Download a GRIB2 file from *url* and save it to *local_path*.

        Retries up to :data:`_MAX_RETRIES` times with a
        :data:`_RETRY_SLEEP_S`-second sleep between attempts.

        Parameters
        ----------
        url : str
            Remote GRIB2 URL to download.
        local_path : Path
            Destination file path (parent directory must exist).

        Returns
        -------
        Path
            *local_path* on successful download.

        Raises
        ------
        IngestionError
            When all retry attempts fail.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(url) as response:
                    local_path.write_bytes(response.read())
                logger.info(
                    "Downloaded GFS GRIB (attempt %d/%d): %s",
                    attempt, _MAX_RETRIES, local_path.name,
                )
                return local_path
            except urllib.error.URLError as exc:
                logger.warning(
                    "GFS download attempt %d/%d failed for %s: %s",
                    attempt, _MAX_RETRIES, url, exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP_S)

        raise IngestionError(
            f"Failed to download GFS GRIB after {_MAX_RETRIES} attempts: {url}"
        )

    # ------------------------------------------------------------------
    # GRIB parsing
    # ------------------------------------------------------------------

    def _parse_grib(self, grib_path: Path) -> xr.Dataset:
        """Parse a single GFS GRIB2 file into a partial contract dataset.

        Steps
        -----
        1. Open all message groups with ``cfgrib.open_datasets``.
        2. Map GFS variable names to contract names via :attr:`GFS_VAR_MAP`.
        3. Rename ``latitude``/``longitude`` coordinates to ``lat``/``lon``.
        4. Crop to the India domain.
        5. Convert ``tp`` from kg m⁻² s⁻¹ → mm hr⁻¹ (multiply by 3600).

        Parameters
        ----------
        grib_path : Path
            Path to the locally downloaded GRIB2 file.

        Returns
        -------
        xr.Dataset
            Dataset with contract variable names, India domain, and
            corrected units.  The ``lead_hours`` dimension is *not* added
            here — that is done in :meth:`ingest`.

        Raises
        ------
        IngestionError
            When none of the expected GFS variable names are found.
        """
        import cfgrib  # Lazy import — requires eccodes system library

        raw_datasets: list[xr.Dataset] = cfgrib.open_datasets(str(grib_path))

        # Collect contract-named DataArrays from across all cfgrib subsets
        contract_arrays: dict[str, xr.DataArray] = {}
        for raw_ds in raw_datasets:
            for gfs_var, contract_var in self.GFS_VAR_MAP.items():
                if gfs_var in raw_ds.data_vars and contract_var not in contract_arrays:
                    contract_arrays[contract_var] = raw_ds[gfs_var]

        if not contract_arrays:
            raise IngestionError(
                f"No recognizable GFS variables found in {grib_path}. "
                f"Expected one of: {list(self.GFS_VAR_MAP.keys())}."
            )

        ds = xr.Dataset(contract_arrays)

        # Rename cfgrib coordinate names to contract names
        rename_map: dict[str, str] = {}
        if "latitude" in ds.coords:
            rename_map["latitude"] = "lat"
        if "longitude" in ds.coords:
            rename_map["longitude"] = "lon"
        if rename_map:
            ds = ds.rename(rename_map)

        # Crop to India domain — handle both ascending and descending lat
        lat_vals = ds.coords["lat"].values
        lat_slice = (
            slice(_LAT_NORTH, _LAT_SOUTH)  # descending (common in GRIB)
            if lat_vals[0] > lat_vals[-1]
            else slice(_LAT_SOUTH, _LAT_NORTH)  # ascending
        )
        ds = ds.sel(lat=lat_slice, lon=slice(_LON_WEST, _LON_EAST))

        # Unit conversion: PRATE kg m⁻² s⁻¹ → mm hr⁻¹
        if "tp" in ds:
            ds["tp"] = ds["tp"] * 3600.0
            ds["tp"].attrs["units"] = "mm/hr"
            logger.debug("Converted PRATE: kg/m²/s → mm/hr.")

        return ds

    # ------------------------------------------------------------------
    # Orchestrated ingest
    # ------------------------------------------------------------------

    def ingest(
        self,
        date: pd.Timestamp,
        lead_hours: list[int] | None = None,
        *,
        run_hour: int = 0,
    ) -> xr.Dataset:
        """Download, parse, and merge GFS forecasts for all *lead_hours*.

        For each lead hour this method:
        1. Builds the NOMADS URL via :meth:`_build_url`.
        2. Downloads the GRIB2 file via :meth:`_download_grib`.
        3. Parses it via :meth:`_parse_grib`.
        4. Tags the slice with its ``lead_hours`` coordinate.

        All lead-hour slices are concatenated and validated before returning.

        Parameters
        ----------
        date : pd.Timestamp
            Forecast initialisation date.
        lead_hours : list[int] | None, optional
            Lead hours to fetch.  Defaults to ``[0, 24, 48, 72, 96, 120]``.
        run_hour : int, keyword-only
            GFS model run hour (0, 6, 12, or 18).  Defaults to ``0``.

        Returns
        -------
        xr.Dataset
            Contract-compliant merged dataset covering all *lead_hours*.

        Raises
        ------
        IngestionError
            On any download failure or missing GRIB variable.
        DataContractError
            If the merged dataset fails :meth:`validate_output`.
        """
        if lead_hours is None:
            lead_hours = [0, 24, 48, 72, 96, 120]

        raw_root = Path(self.config["paths"]["raw_data"])
        grib_dir = raw_root / "gfs" / date.strftime("%Y%m%d")
        grib_dir.mkdir(parents=True, exist_ok=True)

        lead_datasets: list[xr.Dataset] = []
        for lh in lead_hours:
            url = self._build_url(date, run_hour, lh)
            local_path = (
                grib_dir / f"gfs.t{run_hour:02d}z.pgrb2.0p25.f{lh:03d}.grib2"
            )
            self._download_grib(url, local_path)

            lh_ds = self._parse_grib(local_path)
            lh_ds = lh_ds.expand_dims({"lead_hours": [lh]})
            lead_datasets.append(lh_ds)
            logger.debug(
                "Parsed GFS lead_hour=%d for %s.", lh, f"{date:%Y-%m-%d}"
            )

        merged: xr.Dataset = xr.concat(lead_datasets, dim="lead_hours")
        merged.attrs.update(
            {
                "model_name": "gfs",
                "model_type": "nwp",
                "units": str(
                    {
                        "t2m": "K",
                        "tp": "mm/hr",
                        "u10": "m/s",
                        "v10": "m/s",
                        "mslp": "Pa",
                    }
                ),
            }
        )

        self.validate_output(merged)
        logger.info(
            "GFS ingest complete: %s, run=%02dZ, %d lead hours.",
            f"{date:%Y-%m-%d}",
            run_hour,
            len(lead_hours),
        )
        return merged
