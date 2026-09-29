import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent

import scripts.generate_demo_data as gdd
from src.utils.config_loader import load_config
from src.ingestion.demo_ingestor import DemoIngestor
from src.preprocessing.variable_mapper import VariableMapper
from src.preprocessing.regridder import Regridder
from src.preprocessing.quality_control import QualityControl
from src.skill.skill_computer import SkillComputer
from src.skill.weight_generator import compute_weights

# 1. Generate data
_START = date(2024, 6, 1)
_END = date(2024, 6, 2)
gdd.generate(_START, _END)

# 2. Configs
pipeline_cfg = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
models_cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
regions_cfg = load_config(str(PROJECT_ROOT / "config" / "regions.yaml"))

_DEMO_MODELS = {"pangu": "ai", "graphcast": "ai", "ecmwf": "nwp", "gfs": "nwp"}

forecast_archive = {k: [] for k in _DEMO_MODELS}
truth_archive = []

mapper = VariableMapper()
regridder = Regridder()
qc = QualityControl()

# Ingest and preprocess models
for model_key, model_type in _DEMO_MODELS.items():
    model_config = {"name": model_key, "type": model_type, "source": "demo", "key": model_key, "var_mapping": {}, "source_units": {}}
    if model_key in models_cfg.get("models", {}):
        model_config.update(models_cfg["models"][model_key])
        
    ingestor = DemoIngestor(pipeline_cfg, model_config)
    
    for i in range((_END - _START).days + 1):
        day = _START + timedelta(days=i)
        ts = pd.Timestamp(day.strftime("%Y-%m-%d"))
        
        ds = ingestor.ingest(ts)
        ds = mapper.map(ds, model_key, {"models": {model_key: model_config}})
        ds = regridder.regrid(ds)
        ds = qc.check(ds)
        ds = ds.assign_coords({"valid_time": ts})
        
        forecast_archive[model_key].append(ds)

# Load truth directly
obs_dir = PROJECT_ROOT / pipeline_cfg["paths"]["raw_data"] / "observations"
for i in range((_END - _START).days + 1):
    day = _START + timedelta(days=i)
    ts = pd.Timestamp(day.strftime("%Y-%m-%d"))
    date_str = day.strftime("%Y%m%d")
    
    t_ds = xr.open_dataset(obs_dir / f"truth_{date_str}.nc")
    t_ds = mapper.map(t_ds, "truth", {"models": {"truth": {"var_mapping": {}, "source_units": {}}}})
    t_ds = regridder.regrid(t_ds)
    t_ds = qc.check(t_ds)
    t_ds = t_ds.assign_coords({"valid_time": ts})
    truth_archive.append(t_ds)


computer = SkillComputer(config=pipeline_cfg, regions=regions_cfg)
skill_df = computer.compute(forecast_archive, truth_archive)

final_df = compute_weights(skill_df, method="inverse_rmse")

pangu_rmse = final_df[final_df["model"] == "pangu"]["rmse"].mean()
gfs_rmse = final_df[final_df["model"] == "gfs"]["rmse"].mean()
print(f"Mean RMSE - Pangu: {pangu_rmse:.4f}, GFS: {gfs_rmse:.4f}")

lowest = final_df.groupby("model")["rmse"].mean().idxmin()
print("Model with lowest RMSE:", lowest)

group_sums = final_df.groupby(["variable", "region", "lead_hours", "season"])["weight"].sum()
all_one = np.allclose(group_sums.values, 1.0)
print(f"Do all weights sum to 1.0? {all_one}")
