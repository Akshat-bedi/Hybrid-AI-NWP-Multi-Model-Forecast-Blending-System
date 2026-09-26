"""
src/blending/inverse_rmse_blender.py
------------------------------------
Layer   : blending
Purpose : Baseline blender utilizing inverse-RMSE weights.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from src.blending.base_blender import BaseBlender
from src.ingestion.base_ingestor import BaseIngestor

logger = logging.getLogger(__name__)


class InverseRMSEBlender(BaseBlender):
    """Blends NWP and AI forecasts using spatially-varying inverse-RMSE weights.
    
    This is the highly robust baseline blender. It guarantees an output 
    irrespective of missing data by falling back to equal weighting.
    """

    _VARIABLES = ["t2m", "tp", "u10", "v10", "mslp"]

    def blend(
        self,
        model_datasets: dict[str, xr.Dataset],
        weights: pd.DataFrame,
        region_masks: dict[str, np.ndarray],
        meta: dict
    ) -> xr.Dataset:
        """Blend datasets across physical regions and lead hours."""
        season = meta.get("season", "Unknown")
        
        # Grab a reference dataset to extract coordinates
        ref_ds = next(iter(model_datasets.values()))
        lead_hours = ref_ds.coords["lead_hours"].values
        lats = ref_ds.coords["lat"].values
        lons = ref_ds.coords["lon"].values
        shape = (len(lead_hours), len(lats), len(lons))
        
        # Determine the definitive region for every grid point. 
        # Base is "all_india", then overwritten by specific bounding boxes.
        region_map = np.full((len(lats), len(lons)), "all_india", dtype=object)
        for region, mask in region_masks.items():
            if region != "all_india":
                region_map[mask] = region
                
        unique_regions = np.unique(region_map)
        model_names = list(model_datasets.keys())
        num_models = len(model_names)
        
        blended_data_vars = {}

        for var in self._VARIABLES:
            blended_var = np.zeros(shape, dtype=np.float32)
            
            for region in unique_regions:
                # 2D boolean mask for the current specific region partition
                r_mask = (region_map == region)
                
                for lh_idx, lh in enumerate(lead_hours):
                    # 2. Look up weights for (variable, region, lead_hours, season)
                    # Use boolean indexing on the DataFrame
                    w_slice = weights.loc[
                        (weights["variable"] == var) &
                        (weights["region"] == region) &
                        (weights["lead_hours"] == lh) &
                        (weights["season"] == season)
                    ]
                    
                    model_weights = {}
                    if not w_slice.empty and len(w_slice) > 0:
                        # Extract weights mapped to model names
                        for _, row in w_slice.iterrows():
                            model_weights[row["model"]] = row["weight"]
                    
                    # Fallback to equal weights if missing / incomplete
                    if not model_weights or len(model_weights) == 0:
                        eq = 1.0 / num_models if num_models > 0 else 1.0
                        model_weights = {m: eq for m in model_names}
                    
                    # Ensure all active models have a weight (fallback equal if a model is missing)
                    for m in model_names:
                        if m not in model_weights:
                            model_weights[m] = 0.0
                    
                    # Normalize just to be perfectly safe after fallbacks
                    total_w = sum(model_weights.values())
                    if total_w == 0:
                        model_weights = {m: 1.0 / num_models for m in model_names}
                    elif not np.isclose(total_w, 1.0):
                        model_weights = {m: w / total_w for m, w in model_weights.items()}

                    # 3. Compute the blend for these specific pixels
                    blend_accum = np.zeros_like(blended_var[lh_idx])
                    
                    for m_name in model_names:
                        w_i = model_weights[m_name]
                        f_i = model_datasets[m_name][var].values[lh_idx]
                        
                        # Add weighted contribution where region is true
                        # (using np.where to avoid modifying pixels outside r_mask)
                        blend_accum = blend_accum + (w_i * f_i)
                        
                    # Splice the accumulated region into the main array
                    blended_var[lh_idx][r_mask] = blend_accum[r_mask]
            
            # Apply boundary smoothing to blend the regional seams smoothly
            blended_var = self._apply_boundary_smoothing(blended_var, sigma=1.0)
            
            blended_data_vars[var] = xr.DataArray(
                blended_var, 
                dims=["lead_hours", "lat", "lon"]
            )

        # Reconstruct into a valid xr.Dataset
        blended_ds = xr.Dataset(
            data_vars=blended_data_vars,
            coords={
                "lead_hours": lead_hours,
                "lat": lats,
                "lon": lons
            },
            attrs={
                "model_name": "hybrid_blend",
                "model_type": "ensemble",
                "blend_method": "inverse_rmse",
                "units": ref_ds.attrs.get("units", "{}")
            }
        )

        from src.ingestion.demo_ingestor import DemoIngestor
        validator = DemoIngestor(config=self.config, model_config={"name": "validator", "type": "ensemble", "key": "val", "var_mapping": {}, "source_units": {}})
        validator.validate_output(blended_ds)

        return blended_ds
