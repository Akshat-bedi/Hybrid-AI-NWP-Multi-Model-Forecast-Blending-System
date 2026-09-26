"""
src/preprocessing/quality_control.py
------------------------------------
Layer   : preprocessing
Purpose : Quality control to flag physically impossible values.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class QualityControl:
    """Flags physically impossible values in the dataset."""

    PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
        "t2m":  (200.0, 340.0),      # K
        "tp":   (0.0, 500.0),        # mm/hr
        "u10":  (-100.0, 100.0),     # m/s
        "v10":  (-100.0, 100.0),     # m/s
        "mslp": (87000.0, 108000.0), # Pa
    }

    def check(self, ds: xr.Dataset) -> xr.Dataset:
        """Check variables against PHYSICAL_BOUNDS and set outliers to NaN.

        Logs a warning if > 5% of grid points are flagged per lead_hour.
        Logs an error if > 30% of grid points are flagged per lead_hour.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to check.

        Returns
        -------
        xr.Dataset
            Dataset with outliers replaced by NaN and QC attributes added.
        """
        out_ds = ds.copy()
        flagged_counts = {}

        for var, (min_val, max_val) in self.PHYSICAL_BOUNDS.items():
            if var not in out_ds.data_vars:
                continue

            da = out_ds[var]
            
            # Find outliers (values outside the valid range)
            outliers = (da < min_val) | (da > max_val)
            
            if "lead_hours" in da.dims:
                for lh in da.coords["lead_hours"].values:
                    slice_outliers = outliers.sel(lead_hours=lh).values
                    count = int(np.sum(slice_outliers))
                    
                    if count > 0:
                        total_pts = slice_outliers.size
                        pct = (count / total_pts) * 100.0
                        
                        flagged_counts[f"{var}_lh{int(lh)}"] = count
                        
                        if pct > 30.0:
                            logger.error(
                                "QC Error: %s at lead_hour %d has %.1f%% outliers (%d/%d pts).",
                                var, int(lh), pct, count, total_pts
                            )
                        elif pct > 5.0:
                            logger.warning(
                                "QC Warning: %s at lead_hour %d has %.1f%% outliers (%d/%d pts).",
                                var, int(lh), pct, count, total_pts
                            )
            else:
                count = int(outliers.sum().item())
                if count > 0:
                    total_pts = outliers.size
                    pct = (count / total_pts) * 100.0
                    flagged_counts[var] = count
                    
                    if pct > 30.0:
                        logger.error("QC Error: %s has %.1f%% outliers.", var, pct)
                    elif pct > 5.0:
                        logger.warning("QC Warning: %s has %.1f%% outliers.", var, pct)

            # Replace outliers with NaN
            out_ds[var] = out_ds[var].where(~outliers, np.nan)

        out_ds.attrs["qc_applied"] = True
        out_ds.attrs["qc_flagged_counts"] = str(flagged_counts)
        return out_ds
