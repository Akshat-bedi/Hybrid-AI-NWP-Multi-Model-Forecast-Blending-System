"""
tests/test_phase4_integration.py
---------------------------------
CHECK 3 — Phase 0 → Phase 4 end-to-end integration test.

Full pipeline sequence exercised via the ForecastOrchestrator:
  1. generate_demo_data.generate()          [Phase 0]
  2. Orchestrator.run() executing:
     a) Ingest enabled models               [Phase 1]
     b) Regrid and QC                       [Phase 2]
     c) Load Skill weights                  [Phase 3 logic]
     d) Compute final blend (RMSE/Booster)  [Phase 4]
     e) Write blended NetCDF artifact       [Phase 4]

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.orchestrator import ForecastOrchestrator
from src.utils.config_loader import load_config

_START = date(2024, 6, 1)
_END = date(2024, 6, 1) # Only need 1 day for orchestrated run

def _make_pipeline_cfg(raw_root: Path, skill_root: Path, out_root: Path) -> dict:
    return {
        "paths": {
            "raw_data": str(raw_root),
            "processed_data": str(raw_root.parent / "processed"),
            "blended_data": str(out_root),
            "skill_scores": str(skill_root),
            "logs": str(raw_root.parent / "logs"),
        },
        "scheduler": {"interval_hours": 6, "run_on_start": True},
        "api": {"host": "0.0.0.0", "port": 8000},
    }


@pytest.fixture(scope="class")
def phase4_orchestrator_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import scripts.generate_demo_data as gdd

    # 1. Setup paths
    base_dir = tmp_path_factory.mktemp("phase4_run")
    raw_root = base_dir / "raw"
    skill_root = base_dir / "skill"
    out_root = base_dir / "blended"
    
    raw_root.mkdir()
    skill_root.mkdir()
    out_root.mkdir()
    
    # 2. Setup Configs
    real_blend_cfg = load_config(str(PROJECT_ROOT / "config" / "blend_config.yaml"))
    real_models_cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
    real_regions_cfg = load_config(str(PROJECT_ROOT / "config" / "regions.yaml"))
    tmp_pipeline_cfg = _make_pipeline_cfg(raw_root, skill_root, out_root)

    def _mock_load_config(path: str) -> dict:
        if "blend_config" in path:
            return real_blend_cfg
        elif "models.yaml" in path:
            return real_models_cfg
        elif "regions.yaml" in path:
            return real_regions_cfg
        return tmp_pipeline_cfg

    # 3. Generate Phase 0 Data
    with patch.object(gdd, "load_config", side_effect=_mock_load_config):
        gdd.generate(start=_START, end=_END)

    # 4. Generate a mock weights parquet file (Simulating Phase 3 previous execution)
    weights_df = pd.DataFrame([
        {"model": "pangu", "variable": "t2m", "region": "all_india", "lead_hours": 0, "season": "JJA", "weight": 0.8},
        {"model": "gfs", "variable": "t2m", "region": "all_india", "lead_hours": 0, "season": "JJA", "weight": 0.2}
    ])
    weights_df.to_parquet(skill_root / "weights_20240601.parquet")

    # 5. Run Orchestrator
    init_time = pd.Timestamp("2024-06-01")
    
    # We patch load_config inside orchestrator to use our mocked paths
    with patch("src.pipeline.orchestrator.load_config", side_effect=_mock_load_config):
        orchestrator = ForecastOrchestrator(
            config=tmp_pipeline_cfg, 
            model_config=real_models_cfg, 
            regions=real_regions_cfg
        )
        blended_ds = orchestrator.run(init_time)
        
    out_path = out_root / f"{init_time:%Y%m%dT%H}_blended.nc"
    return out_path


class TestPhase4Integration:

    def test_pipeline_produces_blended_artifact(self, phase4_orchestrator_output: Path) -> None:
        """Verifies the Orchestrator ran end-to-end and saved a valid NetCDF file."""
        assert phase4_orchestrator_output.exists()
        
    def test_blended_artifact_is_fully_compliant(self, phase4_orchestrator_output: Path) -> None:
        """Verifies the final artifact contains correct dimensions and blending attributes."""
        ds = xr.open_dataset(phase4_orchestrator_output)
        
        # Check Contract
        assert set(ds.dims) == {"lead_hours", "lat", "lon"}
        assert all(v in ds for v in ["t2m", "tp", "u10", "v10", "mslp"])
        assert ds.attrs.get("model_name") == "hybrid_blend"
        assert ds.attrs.get("blend_method") == "inverse_rmse"
        
        # Check Extreme Booster features
        assert "extreme_flags" in ds
        assert "alert_level" in ds
        
        ds.close()
