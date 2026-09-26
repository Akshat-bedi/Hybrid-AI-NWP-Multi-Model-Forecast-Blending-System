"""
src/blending/extreme_booster.py
-------------------------------
Layer   : blending
Purpose : Extreme weather event booster and alert flag generator.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class ExtremeBooster:
    """Detects severe weather signatures and re-evaluates blending weights.
    
    Enhances precipitation forecasting during extreme events by dynamically
    falling back to ETS (Equitable Threat Score) derived weights, heavily
    favoring models with historical accuracy at extreme thresholds.
    """

    def __init__(self, blend_config: dict) -> None:
        """Initialize the booster with threshold parameters."""
        self.config = blend_config
        
        # Load thresholds (applying standard defaults if missing)
        thresholds = self.config.get("thresholds", {})
        self.heavy_rain_mmhr = thresholds.get("heavy_rain_mmday", 64.5) / 24.0
        self.heatwave_k = thresholds.get("heatwave_celsius", 40.0) + 273.15
        self.high_wind_ms = thresholds.get("high_wind_ms", 15.0)

    def apply(
        self,
        blended_ds: xr.Dataset,
        model_datasets: dict[str, xr.Dataset],
        skill_df: pd.DataFrame
    ) -> xr.Dataset:
        """Apply extreme boosting logic to the blended dataset.
        
        Evaluates heavy rain, heatwave, and extreme wind signatures. Overwrites
        blended precipitation values with ETS-focused weights where extreme 
        rain is detected by ANY underlying model. Injects alert flag bitmasks.
        """
        out_ds = blended_ds.copy()
        model_names = list(model_datasets.keys())
        
        # ---------------------------------------------------------
        # Step 1a: Identify Extreme Rain (ANY model exceeds threshold)
        # ---------------------------------------------------------
        # Aggregate the maximum precipitation forecasted across all models
        max_tp_any_model = np.zeros_like(out_ds["tp"].values)
        for m_name, ds in model_datasets.items():
            if "tp" in ds.data_vars:
                max_tp_any_model = np.maximum(max_tp_any_model, ds["tp"].values)
                
        extreme_mask_rain = (max_tp_any_model > self.heavy_rain_mmhr)
        
        if np.any(extreme_mask_rain):
            logger.info("Extreme rain signatures detected! Re-blending affected pixels via ETS weights.")
            tp_blended_arr = out_ds["tp"].values.copy()
            lead_hours = out_ds.coords["lead_hours"].values
            
            # Since pixel-level ETS might be complex to align with regions instantly, 
            # we'll compute global (all_india) ETS weights per lead hour.
            for lh_idx, lh in enumerate(lead_hours):
                lh_mask = extreme_mask_rain[lh_idx]
                if not np.any(lh_mask):
                    continue
                    
                # Look up ETS scores for 'tp' at this lead hour (defaulting to all_india)
                w_slice = skill_df.loc[
                    (skill_df["variable"] == "tp") & 
                    (skill_df["lead_hours"] == lh) &
                    (skill_df["region"] == "all_india") # fallback broad region
                ]
                
                ets_weights = {m: 0.0 for m in model_names}
                if not w_slice.empty:
                    # ETS can be negative. Floor it at 0.0 before weighting.
                    scores = {}
                    for _, row in w_slice.iterrows():
                        scores[row["model"]] = max(0.0, row.get("ets_64.5", 0.0))
                        
                    total_ets = sum(scores.values())
                    if total_ets > 0:
                        ets_weights = {m: s / total_ets for m, s in scores.items()}
                    else:
                        ets_weights = {m: 1.0 / len(model_names) for m in model_names}
                else:
                    ets_weights = {m: 1.0 / len(model_names) for m in model_names}
                    
                # Compute replacement blended precipitation for this lead hour
                reblended_tp = np.zeros_like(tp_blended_arr[lh_idx])
                for m_name in model_names:
                    w_i = ets_weights.get(m_name, 0.0)
                    if "tp" in model_datasets[m_name].data_vars:
                        f_i = model_datasets[m_name]["tp"].values[lh_idx]
                        reblended_tp += w_i * f_i
                        
                # Overwrite only the extreme pixels
                tp_blended_arr[lh_idx][lh_mask] = reblended_tp[lh_mask]
                
            out_ds["tp"].values = tp_blended_arr

        # ---------------------------------------------------------
        # Step 1b: Identify Extreme Heat (from final blend)
        # ---------------------------------------------------------
        extreme_mask_heat = (out_ds["t2m"].values > self.heatwave_k)

        # ---------------------------------------------------------
        # Step 1c: Identify Extreme Wind (from final blend)
        # ---------------------------------------------------------
        wind_speed = np.sqrt(out_ds["u10"].values**2 + out_ds["v10"].values**2)
        extreme_mask_wind = (wind_speed > self.high_wind_ms)

        # ---------------------------------------------------------
        # Step 2: Create alert flags (bitmask: 1=rain, 2=heat, 4=wind)
        # ---------------------------------------------------------
        flags = np.zeros_like(out_ds["t2m"].values, dtype=int)
        flags += 1 * extreme_mask_rain
        flags += 2 * extreme_mask_heat
        flags += 4 * extreme_mask_wind

        # ---------------------------------------------------------
        # Step 3: Create alert_level (0=green, 1=amber, 2=red)
        # ---------------------------------------------------------
        alert_level = np.zeros_like(flags, dtype=int)
        
        # Any threshold exceeded -> Level 1 (Amber)
        alert_level[flags > 0] = 1
        
        # Extreme thresholds -> Level 2 (Red)
        # Rain > 2*threshold OR Wind > 1.5*threshold
        red_mask_rain = (out_ds["tp"].values > 2.0 * self.heavy_rain_mmhr)
        red_mask_wind = (wind_speed > 1.5 * self.high_wind_ms)
        alert_level[red_mask_rain | red_mask_wind] = 2

        # Assign to dataset
        out_ds["extreme_flags"] = xr.DataArray(flags, dims=["lead_hours", "lat", "lon"])
        out_ds["alert_level"] = xr.DataArray(alert_level, dims=["lead_hours", "lat", "lon"])
        out_ds["wind_speed"] = xr.DataArray(wind_speed, dims=["lead_hours", "lat", "lon"])

        return out_ds
