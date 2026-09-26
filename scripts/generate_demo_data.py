"""
scripts/generate_demo_data.py
-----------------------------
Purpose : Generate synthetic demo forecast and observation NetCDF files that
          satisfy the project data contract for offline development and testing.

Usage
-----
  python scripts/generate_demo_data.py --start 2024-06-01 --end 2024-08-31

The script:
  1. Loads ERA5 truth from data/raw/observations/era5_JJA2024_india.nc when
     available.  Falls back to a physically-plausible synthetic base field
     if the file is absent.
  2. Applies per-model bias + Gaussian noise (parameters read from
     config/blend_config.yaml) to produce 4 model output files per day.
  3. Saves every file conforming to the Sacred Data Contract:
       Variables  : t2m (K), tp (mm/hr), u10 (m/s), v10 (m/s), mslp (Pa)
       Dimensions : [lead_hours, lat, lon]
       Coords     : lead_hours=[0,24,48,72,96,120]
                    lat=np.arange(6.0,38.25,0.25) (117 pts)
                    lon=np.arange(68.0,97.25,0.25) (117 pts)
       Attrs      : model_name, model_type, units

By design the noise_std values in blend_config.yaml guarantee:
  pangu RMSE < ecmwf RMSE < gfs RMSE < graphcast RMSE

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import date, timedelta
from typing import Dict

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Ensure the project root is importable so config_loader resolves correctly
# when the script is executed from any working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging — never print()
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("generate_demo_data")

# ---------------------------------------------------------------------------
# Sacred Data Contract constants
# ---------------------------------------------------------------------------
LEAD_HOURS: list[int] = [0, 24, 48, 72, 96, 120]
LAT: np.ndarray = np.arange(6.0, 38.25, 0.25)   # 129 points
LON: np.ndarray = np.arange(68.0, 97.25, 0.25)  # 117 points

# shape reference: (n_leads, n_lat, n_lon)
SHAPE = (len(LEAD_HOURS), len(LAT), len(LON))

# Model type look-up (matches models.yaml)
MODEL_TYPES: Dict[str, str] = {
    "gfs": "nwp",
    "ecmwf": "nwp",
    "pangu": "ai",
    "graphcast": "ai",
}

UNITS: dict = {
    "t2m": "K",
    "tp": "mm/hr",
    "u10": "m/s",
    "v10": "m/s",
    "mslp": "Pa",
}


# ---------------------------------------------------------------------------
# Helper — build grid arrays
# ---------------------------------------------------------------------------

def _build_lat_lon_grids() -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_grid, lon_grid) broadcast to SHAPE for vectorised ops.

    Returns
    -------
    lat_grid : np.ndarray  shape (n_leads, n_lat, n_lon)
    lon_grid : np.ndarray  shape (n_leads, n_lat, n_lon)
    """
    lat_2d, lon_2d = np.meshgrid(LAT, LON, indexing="ij")
    lat_grid = np.broadcast_to(lat_2d[np.newaxis, :, :], SHAPE).copy()
    lon_grid = np.broadcast_to(lon_2d[np.newaxis, :, :], SHAPE).copy()
    return lat_grid, lon_grid


# ---------------------------------------------------------------------------
# Synthetic base-truth generator
# ---------------------------------------------------------------------------

def _generate_synthetic_base(day: date) -> Dict[str, np.ndarray]:
    """Create a physically-plausible synthetic base truth for *day*.

    The base field is deterministic per day (seeded by day-of-year) so that
    the day-to-day sequence is smooth and reproducible.

    Parameters
    ----------
    day : date
        Calendar date for which to generate the base field.

    Returns
    -------
    dict
        Keys: 't2m', 'tp', 'u10', 'v10', 'mslp'
        Values: np.ndarray of shape SHAPE
    """
    doy = day.timetuple().tm_yday          # 1-366
    seasonal_factor = np.sin(2 * np.pi * doy / 365.0)  # -1 … +1

    lat_grid, lon_grid = _build_lat_lon_grids()

    # --- t2m: 300 K + latitudinal gradient + seasonal offset (K) ---
    t2m_base = (
        300.0
        + 5.0 * np.sin(np.deg2rad(lat_grid))
        + 3.0 * seasonal_factor
    )

    # --- tp: log-normal with sparse rain events (mm/hr) ---
    rng = np.random.default_rng(seed=doy * 17 + 31)
    rain_mask = rng.random(SHAPE) < 0.15        # ~15 % of cells have rain
    tp_base = np.where(
        rain_mask,
        rng.lognormal(mean=-1.0, sigma=1.2, size=SHAPE),
        0.0,
    )

    # --- u10/v10: Gaussian centred at 3 m/s (m/s) ---
    u10_base = rng.normal(loc=3.0, scale=1.5, size=SHAPE)
    v10_base = rng.normal(loc=3.0, scale=1.5, size=SHAPE)

    # --- mslp: background pressure + meridional gradient (Pa) ---
    mslp_base = (
        101_325.0
        - 5.0 * (lat_grid - 20.0)      # pressure decreases northward in monsoon
        + 2.0 * (lon_grid - 82.5)      # slight W→E gradient
    )

    return {
        "t2m": t2m_base.astype(np.float32),
        "tp": tp_base.astype(np.float32),
        "u10": u10_base.astype(np.float32),
        "v10": v10_base.astype(np.float32),
        "mslp": mslp_base.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# ERA5 truth loader
# ---------------------------------------------------------------------------

def _load_era5_base(era5_path: Path, day: date) -> Dict[str, np.ndarray] | None:
    """Attempt to load ERA5 data and return base-truth arrays for *day*.

    If the file cannot be read or the required variables are absent this
    function returns ``None`` and the caller falls back to the synthetic
    generator.

    Parameters
    ----------
    era5_path : Path
        Path to the ERA5 NetCDF file.
    day : date
        Calendar date to slice from the ERA5 dataset.

    Returns
    -------
    dict or None
        Variable arrays of shape SHAPE, or None on failure.
    """
    if not era5_path.exists():
        return None

    try:
        ds = xr.open_dataset(era5_path)
        day_str = day.strftime("%Y-%m-%d")
        if "time" in ds.dims:
            ds = ds.sel(time=day_str, method="nearest")
        required = {"t2m", "tp", "u10", "v10", "mslp"}
        if not required.issubset(set(ds.data_vars)):
            logger.warning(
                "ERA5 file missing variables %s — using synthetic base.",
                required - set(ds.data_vars),
            )
            return None

        # Broadcast / reshape to SHAPE (lead_hours, lat, lon)
        def _expand(arr: np.ndarray) -> np.ndarray:
            """Tile a 2-D field across lead-hour dimension."""
            while arr.ndim < 2:
                arr = arr[np.newaxis, :]
            return np.broadcast_to(arr[np.newaxis, :, :], SHAPE).copy().astype(np.float32)

        return {
            "t2m": _expand(ds["t2m"].values),
            "tp": _expand(ds["tp"].values),
            "u10": _expand(ds["u10"].values),
            "v10": _expand(ds["v10"].values),
            "mslp": _expand(ds["mslp"].values),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load ERA5 data (%s) — using synthetic base.", exc)
        return None


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_dataset(
    arrays: Dict[str, np.ndarray],
    model_name: str,
    model_type: str,
) -> xr.Dataset:
    """Wrap variable arrays into an xr.Dataset conforming to the data contract.

    Parameters
    ----------
    arrays : dict
        Variable name → np.ndarray of shape SHAPE.
    model_name : str
        Short identifier used in the ``model_name`` attribute.
    model_type : str
        One of ``"nwp"``, ``"ai"``, or ``"ensemble"``.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions [lead_hours, lat, lon] and required attrs.
    """
    coords = {
        "lead_hours": LEAD_HOURS,
        "lat": LAT,
        "lon": LON,
    }
    data_vars = {
        var: xr.DataArray(data, dims=["lead_hours", "lat", "lon"], coords=coords)
        for var, data in arrays.items()
    }
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            "model_name": model_name,
            "model_type": model_type,
            "units": str(UNITS),  # xarray attrs must be JSON-serialisable
        },
    )


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate(start: date, end: date) -> None:
    """Generate demo forecast and truth files for every day in [start, end].

    Parameters
    ----------
    start : date
        First day of the generation range (inclusive).
    end : date
        Last day of the generation range (inclusive).
    """
    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    blend_cfg = load_config(str(PROJECT_ROOT / "config" / "blend_config.yaml"))
    pipeline_cfg = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))

    demo_params: Dict[str, dict] = blend_cfg["demo_models"]
    raw_root = PROJECT_ROOT / pipeline_cfg["paths"]["raw_data"]
    obs_dir = raw_root / "observations"
    era5_path = obs_dir / "era5_JJA2024_india.nc"

    # ------------------------------------------------------------------
    # Iterate over days
    # ------------------------------------------------------------------
    total_days = (end - start).days + 1
    file_count = 0

    for day_offset in range(total_days):
        day = start + timedelta(days=day_offset)
        date_str = day.strftime("%Y%m%d")
        logger.info("Processing day %s …", date_str)

        # ---- Base truth -----------------------------------------------
        base_arrays = _load_era5_base(era5_path, day)
        if base_arrays is None:
            logger.info("  ERA5 not available — generating synthetic base for %s.", date_str)
            base_arrays = _generate_synthetic_base(day)

        # ---- Save truth -----------------------------------------------
        truth_path = obs_dir / f"truth_{date_str}.nc"
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_ds = _build_dataset(base_arrays, model_name="truth", model_type="ensemble")
        truth_ds.to_netcdf(truth_path)
        logger.debug("  Saved truth → %s", truth_path)

        # ---- Per-model perturbed outputs -------------------------------
        for model_name, model_type in MODEL_TYPES.items():
            params = demo_params[model_name]
            noise_std: float = params["noise_std"]
            bias_t2m: float = params["bias_t2m"]
            bias_tp: float = params["bias_tp"]

            rng = np.random.default_rng(
                seed=hash((day_offset, model_name)) & 0xFFFF_FFFF
            )

            def _perturb(arr: np.ndarray, bias: float = 0.0) -> np.ndarray:
                """Add bias + Gaussian noise; return float32 array."""
                return (arr + bias + rng.normal(0.0, noise_std, arr.shape)).astype(np.float32)

            model_arrays = {
                "t2m": _perturb(base_arrays["t2m"], bias=bias_t2m),
                "tp": np.clip(
                    _perturb(base_arrays["tp"], bias=bias_tp), 0.0, None
                ),  # precipitation non-negative
                "u10": _perturb(base_arrays["u10"]),
                "v10": _perturb(base_arrays["v10"]),
                "mslp": _perturb(base_arrays["mslp"]),
            }

            model_dir = raw_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{date_str}.nc"
            model_ds = _build_dataset(model_arrays, model_name=model_name, model_type=model_type)
            model_ds.to_netcdf(model_path)
            logger.debug("  Saved %s → %s", model_name, model_path)
            file_count += 1

    # ------------------------------------------------------------------
    # Summary — logged only (project rule: never use print())
    # ------------------------------------------------------------------
    summary = (
        f"Generated {total_days} days × 4 models = {file_count} files "
        f"(+ {total_days} truth files) in {raw_root}"
    )
    logger.info(summary)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``start`` and ``end`` as ``datetime.date``.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo forecast data for the Hybrid AI-NWP system.",
    )
    parser.add_argument(
        "--start",
        required=True,
        type=lambda s: date.fromisoformat(s),
        help="Start date in YYYY-MM-DD format (inclusive).",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda s: date.fromisoformat(s),
        help="End date in YYYY-MM-DD format (inclusive).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.end < args.start:
        logger.error("--end date must be >= --start date.")
        sys.exit(1)
    generate(start=args.start, end=args.end)
