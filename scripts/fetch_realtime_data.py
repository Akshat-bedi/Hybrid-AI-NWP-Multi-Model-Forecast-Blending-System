"""
scripts/fetch_realtime_data.py
------------------------------
Fetches real-time weather data from the free Open-Meteo API.
Implements batched fetching, parsing into xr.Dataset, and regridding.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import sys
import logging
import math
import time
from pathlib import Path
from datetime import datetime
from itertools import islice

import requests
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.regridder import Regridder
from src.pipeline.orchestrator import ForecastOrchestrator
from src.utils.config_loader import load_config

def batched(iterable, n):
    """Batch data into tuples of length n."""
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch

def fetch_open_meteo_batched(lats, lons):
    """Fetch Open-Meteo API in batches of 50."""
    url = "https://api.open-meteo.com/v1/forecast"
    all_responses = []
    
    # Create lat-lon pairs
    pairs = []
    for lat in lats:
        for lon in lons:
            pairs.append((lat, lon))
            
    # Open-meteo max locations per request is up to 100 on some endpoints, let's try 100
    batch_size = 100
    for i, batch in enumerate(batched(pairs, batch_size)):
        batch_lats = ",".join(str(p[0]) for p in batch)
        batch_lons = ",".join(str(p[1]) for p in batch)
        
        params = {
            "latitude": batch_lats,
            "longitude": batch_lons,
            "hourly": "temperature_2m,precipitation,windspeed_10m,winddirection_10m,surface_pressure",
            "forecast_days": 6,
            "timezone": "Asia/Kolkata",
            "wind_speed_unit": "ms"
        }
        
        logging.info(f"Fetching batch {i+1} / {math.ceil(len(pairs) / batch_size)} ...")
        
        max_retries = 10
        base_delay = 10.0
        
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    delay = base_delay * (1.5 ** attempt)
                    logging.warning(f"429 Too Many Requests. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                delay = base_delay * (1.5 ** attempt)
                logging.warning(f"Request failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
        else:
            raise RuntimeError(f"Failed to fetch batch {i+1} after {max_retries} retries due to rate limiting.")
        
        resp_json = r.json()
        if isinstance(resp_json, dict) and "hourly" in resp_json:
            # If batch_size was 1 or only 1 location returned
            all_responses.append((batch[0][0], batch[0][1], resp_json))
        else:
            for j, loc_resp in enumerate(resp_json):
                all_responses.append((batch[j][0], batch[j][1], loc_resp))
                
        time.sleep(3.0)  # Rate limit protection
        
    return all_responses

def process_responses(responses, lats, lons):
    """Parse batch responses into a 0.5-degree xr.Dataset."""
    lead_indices = [0, 24, 48, 72, 96, 120]
    
    shape = (len(lead_indices), len(lats), len(lons))
    t2m_arr = np.full(shape, np.nan)
    tp_arr = np.full(shape, np.nan)
    u10_arr = np.full(shape, np.nan)
    v10_arr = np.full(shape, np.nan)
    mslp_arr = np.full(shape, np.nan)
    
    # Create fast lookup map for array indexing
    lat_idx_map = {lat: i for i, lat in enumerate(lats)}
    lon_idx_map = {lon: j for j, lon in enumerate(lons)}
    
    for lat, lon, data in responses:
        if "hourly" not in data:
            continue
            
        hourly = data["hourly"]
        
        temp_c = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation", [])
        ws = hourly.get("windspeed_10m", [])
        wd = hourly.get("winddirection_10m", [])
        pres = hourly.get("surface_pressure", [])
        
        i = lat_idx_map.get(lat)
        j = lon_idx_map.get(lon)
        
        if i is None or j is None:
            continue
        
        for t_out, t_in in enumerate(lead_indices):
            if t_in < len(temp_c) and temp_c[t_in] is not None:
                t2m_arr[t_out, i, j] = temp_c[t_in] + 273.15
                
            if t_in < len(precip) and precip[t_in] is not None:
                tp_arr[t_out, i, j] = precip[t_in]
                
            if t_in < len(pres) and pres[t_in] is not None:
                mslp_arr[t_out, i, j] = pres[t_in] * 100.0
                
            if t_in < len(ws) and ws[t_in] is not None and t_in < len(wd) and wd[t_in] is not None:
                speed = ws[t_in]
                direction = wd[t_in]
                u10_arr[t_out, i, j] = speed * math.sin(math.radians(direction))
                v10_arr[t_out, i, j] = speed * math.cos(math.radians(direction))
                
    ds = xr.Dataset(
        data_vars={
            "t2m": (["lead_hours", "lat", "lon"], t2m_arr),
            "tp": (["lead_hours", "lat", "lon"], tp_arr),
            "u10": (["lead_hours", "lat", "lon"], u10_arr),
            "v10": (["lead_hours", "lat", "lon"], v10_arr),
            "mslp": (["lead_hours", "lat", "lon"], mslp_arr),
        },
        coords={
            "lead_hours": [0, 24, 48, 72, 96, 120],
            "lat": lats,
            "lon": lons
        },
        attrs={
            "model_name": "open_meteo",
            "model_type": "nwp",
            "units": str({"t2m": "K", "tp": "mm/hr", "u10": "m/s", "v10": "m/s", "mslp": "Pa"})
        }
    )
    
    return ds

def main():
    logging.info("Starting Open-Meteo real-time data fetch...")
    
    # 1. Define 1.0 grid (drastically reduces API calls)
    lats = np.arange(6.0, 39.0, 1.0)
    lons = np.arange(68.0, 98.0, 1.0)
    
    # 2 & 3. Fetch in batches
    responses = fetch_open_meteo_batched(lats, lons)
    
    # 4 & 5. Build dataset
    logging.info("Parsing responses into Dataset...")
    ds_05deg = process_responses(responses, lats, lons)
    
    # 6. Regrid from 0.5 to 0.25 using existing Regridder
    logging.info("Regridding from 0.5 to 0.25...")
    regridder = Regridder()
    ds_025 = regridder.regrid(ds_05deg)
    
    # 7. Save to data/raw/open_meteo
    today_str = datetime.now().strftime("%Y%m%d")
    out_dir = PROJECT_ROOT / "data" / "raw" / "open_meteo"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"{today_str}.nc"
    ds_025.to_netcdf(out_path)
    logging.info(f"Saved real-time data to {out_path}")
    
    # 8. Run Orchestrator
    config = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
    models_cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
    regions_cfg = load_config(str(PROJECT_ROOT / "config" / "regions.yaml"))
    
    orch = ForecastOrchestrator(config, models_cfg, regions_cfg)
    
    today_ts = pd.Timestamp.now().floor('D')
    logging.info(f"Running Orchestrator for {today_ts}")
    
    orch.run(today_ts)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    print("Real-time data fetched and blended successfully")
