"""
tests/test_phase3_integration.py
---------------------------------
CHECK 3 — Phase 0 → Phase 1 → Phase 2 → Phase 3 end-to-end integration test.

Full pipeline sequence exercised:
  1. load_config()                          [Phase 0]
  2. generate_demo_data.generate()          [Phase 0]
  3. DemoIngestor.ingest()                  [Phase 1]
  4. VariableMapper.map()                   [Phase 2]
  5. Regridder.regrid()                     [Phase 2]
  6. QualityControl.check()                 [Phase 2]
  7. SkillComputer.compute()                [Phase 3]
  8. weight_generator.compute_weights()     [Phase 3]

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.demo_ingestor import DemoIngestor
from src.preprocessing.quality_control import QualityControl
from src.preprocessing.regridder import Regridder
from src.preprocessing.variable_mapper import VariableMapper
from src.skill.skill_computer import SkillComputer
from src.skill.weight_generator import compute_weights
from src.utils.config_loader import load_config

_START = date(2024, 6, 1)
_END = date(2024, 6, 2)
_N_DAYS = (_END - _START).days + 1

_DEMO_MODELS = {
    "pangu": "ai",
    "graphcast": "ai",
    "ecmwf": "nwp",
    "gfs": "nwp",
}


def _make_pipeline_cfg(raw_root: Path) -> dict:
    base = raw_root.parent
    return {
        "paths": {
            "raw_data": str(raw_root),
            "processed_data": str(base / "processed"),
            "blended_data": str(base / "blended"),
            "skill_scores": str(base / "skill"),
            "logs": str(base / "logs"),
        },
        "scheduler": {"interval_hours": 6, "run_on_start": True},
        "api": {"host": "0.0.0.0", "port": 8000},
    }


@pytest.fixture(scope="class")
def phase3_integration_output(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    import scripts.generate_demo_data as gdd

    raw_root = tmp_path_factory.mktemp("phase3_raw")
    real_blend_cfg = load_config(str(PROJECT_ROOT / "config" / "blend_config.yaml"))
    real_models_cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
    
    tmp_pipeline_cfg = _make_pipeline_cfg(raw_root)

    def _mock_load_config(path: str) -> dict:
        if "blend_config" in path:
            return real_blend_cfg
        elif "models.yaml" in path:
            return real_models_cfg
        return tmp_pipeline_cfg

    # 1 & 2. Generate Phase 0 Data
    with patch.object(gdd, "load_config", side_effect=_mock_load_config):
        gdd.generate(start=_START, end=_END)

    forecast_archive: dict[str, list[xr.Dataset]] = {k: [] for k in _DEMO_MODELS}
    truth_archive: list[xr.Dataset] = []

    mapper = VariableMapper()
    regridder = Regridder()
    qc = QualityControl()

    for model_key, model_type in _DEMO_MODELS.items():
        model_config = {
            "name": model_key,
            "type": model_type,
            "source": "demo",
            "key": model_key,
            "var_mapping": {},
            "source_units": {}
        }
        if model_key in real_models_cfg.get("models", {}):
            model_config.update(real_models_cfg["models"][model_key])
            
        ingestor = DemoIngestor(tmp_pipeline_cfg, model_config)
        
        for i in range(_N_DAYS):
            day = _START + timedelta(days=i)
            ts = pd.Timestamp(day.strftime("%Y-%m-%d"))
            
            # Phase 1: Ingest
            ds = ingestor.ingest(ts)
            
            # Phase 2: Preprocess
            ds = mapper.map(ds, model_key, {"models": {model_key: model_config}})
            ds = regridder.regrid(ds)
            ds = qc.check(ds)
            
            # Assigning a time coordinate for season mapping in Phase 3
            ds = ds.assign_coords({"valid_time": ts})
            
            forecast_archive[model_key].append(ds)

    # Mocking Truth Archive (Assume ERA5 is identical to ECMWF for integration test)
    truth_archive = forecast_archive["ecmwf"]

    # Phase 3: Compute Metrics
    regions = load_config(str(PROJECT_ROOT / "config" / "regions.yaml"))
    computer = SkillComputer(config=tmp_pipeline_cfg, regions=regions)
    skill_df = computer.compute(forecast_archive, truth_archive)
    
    # Phase 3: Compute Weights
    final_df = compute_weights(skill_df, method="inverse_rmse")
    return final_df


class TestPhase3Integration:

    def test_pipeline_completes_without_exception(self, phase3_integration_output: pd.DataFrame) -> None:
        assert phase3_integration_output is not None
        assert isinstance(phase3_integration_output, pd.DataFrame)

    def test_all_models_present(self, phase3_integration_output: pd.DataFrame) -> None:
        models = set(phase3_integration_output["model"].unique())
        assert models == set(_DEMO_MODELS.keys())

    def test_no_nan_weights(self, phase3_integration_output: pd.DataFrame) -> None:
        assert phase3_integration_output["weight"].isna().sum() == 0

    def test_weights_sum_to_one_per_group(self, phase3_integration_output: pd.DataFrame) -> None:
        group_sums = phase3_integration_output.groupby(
            ["variable", "region", "lead_hours", "season"]
        )["weight"].sum()
        assert np.allclose(group_sums.values, 1.0)
