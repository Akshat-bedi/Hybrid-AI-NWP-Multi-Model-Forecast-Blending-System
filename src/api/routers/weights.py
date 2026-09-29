"""
src/api/routers/weights.py
--------------------------
Layer   : api
Purpose : Endpoint for mapping regional blending weights based on skill records.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import WeightEntry, WeightMapResponse
from src.utils.config_loader import load_config

router = APIRouter(prefix="/weights", tags=["Weights"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@router.get("/{variable}", response_model=WeightMapResponse)
def get_weights(variable: str, lead_hours: int = 24, season: str = "JJA"):
    """Fetch current regional model weights and colors."""
    try:
        config = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
        models_cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
        
        skill_dir = PROJECT_ROOT / config["paths"]["skill_scores"]
        parquet_files = list(skill_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError()
            
        latest_parquet = max(parquet_files)
        df = pd.read_parquet(latest_parquet)
        
    except Exception:
        raise HTTPException(status_code=404, detail="No weights available")
        
    # Apply strict slice filtering
    mask = (
        (df["variable"] == variable) &
        (df["lead_hours"] == lead_hours) &
        (df["season"] == season)
    )
    df_filtered = df[mask]
    
    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No weights match requested parameters")
        
    regions_dict = {}
    for region_name, group in df_filtered.groupby("region"):
        entries = []
        for _, row in group.iterrows():
            model_name = row["model"]
            color = "#000000"
            if "models" in models_cfg and model_name in models_cfg["models"]:
                color = models_cfg["models"][model_name].get("color", "#000000")
                
            entries.append(WeightEntry(
                model_name=model_name,
                weight=float(row.get("weight", 0.0)),
                rmse=float(row.get("rmse", 0.0)),
                color=color,
                skill_source=row.get("skill_source", "live")
            ))
        regions_dict[str(region_name)] = entries
        
    return WeightMapResponse(
        variable=variable,
        lead_hours=lead_hours,
        season=season,
        regions=regions_dict
    )
