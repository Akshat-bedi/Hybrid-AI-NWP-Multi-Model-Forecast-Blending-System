import sys
from pathlib import Path
from datetime import date
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent

import scripts.generate_demo_data as gdd
from src.utils.config_loader import load_config
from src.pipeline.orchestrator import ForecastOrchestrator

# 1. Generate data
_START = date(2024, 6, 1)
_END = date(2024, 6, 1)
print("Generating demo data...")
gdd.generate(_START, _END)

# 2. Configs
pipeline_cfg = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
models_cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
regions_cfg = load_config(str(PROJECT_ROOT / "config" / "regions.yaml"))

# 3. Create mock parquet weights
skill_dir = PROJECT_ROOT / pipeline_cfg["paths"]["skill_scores"]
skill_dir.mkdir(parents=True, exist_ok=True)
weights_df = pd.DataFrame([
    {"model": "pangu", "variable": var, "region": "all_india", "lead_hours": lh, "season": "JJA", "weight": 0.8}
    for var in ["t2m", "tp", "u10", "v10", "mslp"] for lh in [0, 24, 48, 72, 96, 120]
] + [
    {"model": "gfs", "variable": var, "region": "all_india", "lead_hours": lh, "season": "JJA", "weight": 0.2}
    for var in ["t2m", "tp", "u10", "v10", "mslp"] for lh in [0, 24, 48, 72, 96, 120]
])
weights_df.to_parquet(skill_dir / "weights_20240601.parquet")

# 4. Run Orchestrator
print("Initializing Orchestrator...")
orchestrator = ForecastOrchestrator(
    config=pipeline_cfg, 
    model_config=models_cfg, 
    regions=regions_cfg
)
init_time = pd.Timestamp("2024-06-01")
print(f"Running pipeline for {init_time}...")
blended_ds = orchestrator.run(init_time)

# 5. Verify Output NetCDF
out_path = PROJECT_ROOT / pipeline_cfg["paths"]["blended_data"] / f"{init_time:%Y%m%dT%H}_blended.nc"
print(f"\nVerifying artifact at {out_path}...")

assert out_path.exists(), "Blended NetCDF file does not exist!"

ds = xr.open_dataset(out_path)
print("\nVariables in blended NetCDF:")
for var_name in ds.data_vars:
    print(f" - {var_name}")

required = {"t2m", "tp", "u10", "v10", "mslp", "extreme_flags"}
if required.issubset(set(ds.data_vars)):
    print("\nSUCCESS: Output NetCDF contains all 5 required variables + extreme_flags.")
else:
    print("\nFAILURE: Missing required variables.")

print("\nAttributes:")
for k, v in ds.attrs.items():
    print(f" - {k}: {v}")

print(f"\nAlert levels found: {set(np.unique(ds['alert_level'].values))}")
ds.close()
