"""
src/skill/skill_computer.py
---------------------------
Layer   : skill
Purpose : Compute evaluation metrics across models, regions, and seasons.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from src.skill import metrics

logger = logging.getLogger(__name__)

SEASON_MAP = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11]
}


class SkillComputer:
    """Computes verification metrics over different dimensions (regions, seasons, etc.)."""

    # Contract standard coordinates
    _TARGET_LAT = np.arange(6.0, 38.25, 0.25)
    _TARGET_LON = np.arange(68.0, 97.25, 0.25)

    def __init__(self, config: dict, regions: dict) -> None:
        """Initialize SkillComputer and precompute region masks.
        
        Parameters
        ----------
        config : dict
            Pipeline configuration.
        regions : dict
            Regions configuration (from regions.yaml).
        """
        self.config = config
        self.regions_config = regions
        self.region_masks = self._build_region_masks()

    def _build_region_masks(self) -> dict[str, np.ndarray]:
        """Precompute boolean masks for each defined region."""
        lon_grid, lat_grid = np.meshgrid(self._TARGET_LON, self._TARGET_LAT)
        masks = {}
        
        for region_key, bounds in self.regions_config.get("regions", {}).items():
            mask = (
                (lat_grid >= bounds["lat_min"]) &
                (lat_grid <= bounds["lat_max"]) &
                (lon_grid >= bounds["lon_min"]) &
                (lon_grid <= bounds["lon_max"])
            )
            masks[region_key] = mask
            
        # Add full domain mask
        masks["all_india"] = np.ones_like(lat_grid, dtype=bool)
        
        return masks

    def _get_season(self, month: int) -> str:
        for season, months in SEASON_MAP.items():
            if month in months:
                return season
        return "Unknown"

    def compute(
        self,
        forecast_archive: dict[str, list[xr.Dataset]],
        truth_archive: list[xr.Dataset]
    ) -> pd.DataFrame:
        """Compute verification metrics.
        
        Parameters
        ----------
        forecast_archive : dict[str, list[xr.Dataset]]
            Model forecasts mapped by model name.
        truth_archive : list[xr.Dataset]
            Observation/truth datasets matching the forecast archive index.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all computed metrics.
        """
        results = []
        
        for model, forecasts in forecast_archive.items():
            for i, ds_fcst in enumerate(forecasts):
                ds_truth = truth_archive[i]
                
                # Try to extract the time from dataset (often encoded in attrs or coords, 
                # but for this system we'll check common places, or just use 1 as fallback)
                # In phase 1/2 we typically pass pd.Timestamp into ingest, but xr.Dataset might not have it.
                # Assuming 'valid_time' coordinate or similar, else fallback to Jan.
                # We'll extract from ds_truth or ds_fcst.
                month = 1
                if "time" in ds_fcst.coords:
                    time_val = pd.Timestamp(ds_fcst["time"].values)
                    month = time_val.month
                elif "valid_time" in ds_fcst.coords:
                    time_val = pd.Timestamp(ds_fcst["valid_time"].values)
                    month = time_val.month
                
                season = self._get_season(month)

                # Iterate through all standard variables
                for var in ["t2m", "tp", "u10", "v10", "mslp"]:
                    if var not in ds_fcst.data_vars or var not in ds_truth.data_vars:
                        continue
                        
                    logger.info("Computing skill for %s %s...", model, var)
                    
                    da_fcst = ds_fcst[var].values
                    da_truth = ds_truth[var].values
                    lead_hours = ds_fcst.coords["lead_hours"].values
                    
                    for region, mask in self.region_masks.items():
                        for lh_idx, lh in enumerate(lead_hours):
                            # Extract 2D slices
                            f_slice = da_fcst[lh_idx, :, :]
                            t_slice = da_truth[lh_idx, :, :]
                            
                            # Apply region mask
                            f_masked = f_slice[mask]
                            t_masked = t_slice[mask]
                            
                            r_rmse = metrics.rmse(f_masked, t_masked)
                            r_mae = metrics.mae(f_masked, t_masked)
                            r_bias = metrics.bias(f_masked, t_masked)
                            
                            # Using 64.5 mm/hr as an example threshold for ETS if precipitation
                            ets_64_5 = np.nan
                            if var == "tp":
                                ets_64_5 = metrics.ets(f_masked, t_masked, threshold=64.5)
                            
                            results.append({
                                "model": model,
                                "variable": var,
                                "region": region,
                                "lead_hours": int(lh),
                                "season": season,
                                "rmse": r_rmse,
                                "mae": r_mae,
                                "bias": r_bias,
                                "ets_64.5": ets_64_5
                            })
                            
        return pd.DataFrame(results)

    def save(self, skill_df: pd.DataFrame, path: str) -> None:
        """Save the computed skill DataFrame to disk as a Parquet file."""
        skill_df.to_parquet(path, index=False)
