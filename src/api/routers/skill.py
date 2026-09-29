"""
src/api/routers/skill.py
------------------------
Layer   : api
Purpose : Endpoint for querying raw model evaluation metrics.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import SkillRow
from src.utils.config_loader import load_config

router = APIRouter(prefix="/skill", tags=["Skill"])
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@router.get("/scores", response_model=List[SkillRow])
def get_skill_scores(variable: str = "t2m", lead_hours: int = 72):
    """Fetch raw evaluation metrics from the latest skill compilation."""
    try:
        config = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
        skill_dir = PROJECT_ROOT / config["paths"]["skill_scores"]
        parquet_files = list(skill_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError()
            
        latest_parquet = max(parquet_files)
        df = pd.read_parquet(latest_parquet)
        
    except Exception:
        raise HTTPException(status_code=404, detail="No skill scores available")
        
    mask = (df["variable"] == variable) & (df["lead_hours"] == lead_hours)
    df_filtered = df[mask]
    
    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="No skill scores match requested parameters")
        
    results = []
    for _, row in df_filtered.iterrows():
        # Fallback between potential ETS key names
        ets_val = float(row.get("ets_64.5", row.get("ets", 0.0)))
        
        results.append(SkillRow(
            model=row["model"],
            variable=row["variable"],
            region=row["region"],
            lead_hours=int(row["lead_hours"]),
            rmse=float(row.get("rmse", 0.0)),
            mae=float(row.get("mae", 0.0)),
            bias=float(row.get("bias", 0.0)),
            ets=ets_val,
            skill_source=row.get("skill_source", "live")
        ))
        
    return results
