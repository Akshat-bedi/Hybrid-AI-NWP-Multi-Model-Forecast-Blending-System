"""
src/api/routers/alerts.py
-------------------------
Layer   : api
Purpose : Endpoint for querying extreme weather alerts detected by the booster.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import APIRouter

from src.api.schemas import AlertItem
from src.utils.config_loader import load_config

router = APIRouter(prefix="/alerts", tags=["Alerts"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@router.get("/extreme", response_model=List[AlertItem])
def get_extreme_alerts(valid_time: Optional[str] = None):
    """Fetch active severe weather alerts mapped by region."""
    try:
        config = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
        blended_dir = PROJECT_ROOT / config["paths"]["blended_data"]
        
        nc_files = list(blended_dir.glob("*_blended.nc"))
        if not nc_files:
            return []
            
        file_path = max(nc_files)
        
        with xr.open_dataset(file_path) as ds:
            if "alert_level" not in ds.data_vars or "extreme_flags" not in ds.data_vars:
                return []
                
            regions_cfg = load_config(str(PROJECT_ROOT / "config" / "regions.yaml"))
            defined_regions = regions_cfg.get("regions", {})
            
            lats = ds.coords["lat"].values
            lons = ds.coords["lon"].values
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            
            # Map regions
            region_masks = {}
            for r_name, r_bounds in defined_regions.items():
                m = (
                    (lat_grid >= r_bounds["lat_min"]) & (lat_grid <= r_bounds["lat_max"]) &
                    (lon_grid >= r_bounds["lon_min"]) & (lon_grid <= r_bounds["lon_max"])
                )
                region_masks[r_name] = m
            
            # Fallback if no specific regions define coverage
            region_masks["all_india"] = np.ones_like(lat_grid, dtype=bool)
            
            lead_hours = ds.coords["lead_hours"].values
            
            # Parse initialization time from filename (e.g. 20240601T00)
            init_time_str = file_path.name.split("_")[0]
            try:
                init_time = pd.to_datetime(init_time_str)
            except Exception:
                init_time = pd.Timestamp("2024-01-01")
            
            alerts = []
            
            for lh_idx, lh in enumerate(lead_hours):
                alert_arr = ds["alert_level"].values[lh_idx]
                flags_arr = ds["extreme_flags"].values[lh_idx]
                
                vt = init_time + pd.Timedelta(hours=int(lh))
                vt_str = vt.strftime("%Y-%m-%dT%H:%M:%S")
                
                if valid_time and valid_time != vt_str:
                    continue
                    
                active_mask = (alert_arr >= 1)
                
                if not np.any(active_mask):
                    continue
                    
                # Find which regions are affected
                for r_name, r_mask in region_masks.items():
                    intersection = active_mask & r_mask
                    if not np.any(intersection):
                        continue
                        
                    # Aggregate the alerts for this specific region + time combo
                    max_sev = int(np.max(alert_arr[intersection]))
                    
                    # Bitwise OR across all triggered flags in this region
                    flat_flags = flags_arr[intersection].astype(int)
                    flag_bits = int(np.bitwise_or.reduce(flat_flags))
                    
                    lat_center = float(np.mean(lat_grid[intersection]))
                    lon_center = float(np.mean(lon_grid[intersection]))
                    
                    alert_types = []
                    if flag_bits & 1: alert_types.append("heavy_rain")
                    if flag_bits & 2: alert_types.append("heatwave")
                    if flag_bits & 4: alert_types.append("high_wind")
                    
                    for atype in alert_types:
                        alerts.append(AlertItem(
                            alert_type=atype,
                            severity=max_sev,
                            region=r_name,
                            valid_time=vt_str,
                            lat_center=lat_center,
                            lon_center=lon_center,
                            description=f"{atype.replace('_', ' ').title()} Alert for {r_name}"
                        ))
                        
            # Prevent duplicate regions via a simple dict merge (keeping highest severity)
            # if they fall inside multiple overlapping bounding boxes.
            unique_alerts = {}
            for a in alerts:
                key = (a.alert_type, a.region, a.valid_time)
                if key not in unique_alerts or a.severity > unique_alerts[key].severity:
                    unique_alerts[key] = a
                    
            return list(unique_alerts.values())

    except Exception:
        # Failsafe: Never 404, just return empty list as requested
        return []
