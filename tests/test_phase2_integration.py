"""
tests/test_phase2_integration.py
---------------------------------
CHECK 3 — Phase 0 → Phase 1 → Phase 2 end-to-end integration test.

Full pipeline sequence exercised:
  1. load_config()               [Phase 0]
  2. generate_demo_data.generate()  [Phase 0]
  3. DemoIngestor.ingest()       [Phase 1]
  4. VariableMapper.map()        [Phase 2]
  5. Regridder.regrid()          [Phase 2]
  6. QualityControl.check()      [Phase 2]
  7. BaseIngestor.validate_output() [Phase 1 / Contract check]

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.base_ingestor import BaseIngestor
from src.ingestion.demo_ingestor import DemoIngestor
from src.preprocessing.quality_control import QualityControl
from src.preprocessing.regridder import Regridder
from src.preprocessing.variable_mapper import VariableMapper
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


class _StubForValidation(BaseIngestor):
    def ingest(self, date: pd.Timestamp, lead_hours: list[int]) -> xr.Dataset:
        pass


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
def phase2_pipeline_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[xr.Dataset]]:
    import scripts.generate_demo_data as gdd

    raw_root = tmp_path_factory.mktemp("phase2_raw")
    real_blend_cfg = load_config(str(PROJECT_ROOT / "config" / "blend_config.yaml"))
    real_models_cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
    
    tmp_pipeline_cfg = _make_pipeline_cfg(raw_root)

    def _mock_load_config(path: str) -> dict:
        if "blend_config" in path:
            return real_blend_cfg
        elif "models.yaml" in path:
            return real_models_cfg
        return tmp_pipeline_cfg

    with patch.object(gdd, "load_config", side_effect=_mock_load_config):
        gdd.generate(start=_START, end=_END)

    results: dict[str, list[xr.Dataset]] = {k: [] for k in _DEMO_MODELS}

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
            
            results[model_key].append(ds)

    return results


class TestPhase2Integration:

    def test_pipeline_completes_without_exception(self, phase2_pipeline_results: dict[str, list[xr.Dataset]]) -> None:
        assert phase2_pipeline_results is not None

    def test_all_models_processed(self, phase2_pipeline_results: dict[str, list[xr.Dataset]]) -> None:
        for model_key in _DEMO_MODELS:
            assert model_key in phase2_pipeline_results
            assert len(phase2_pipeline_results[model_key]) == _N_DAYS

    def test_all_results_are_xr_dataset(self, phase2_pipeline_results: dict[str, list[xr.Dataset]]) -> None:
        for datasets in phase2_pipeline_results.values():
            for ds in datasets:
                assert isinstance(ds, xr.Dataset)

    def test_all_results_have_qc_applied(self, phase2_pipeline_results: dict[str, list[xr.Dataset]]) -> None:
        for datasets in phase2_pipeline_results.values():
            for ds in datasets:
                assert ds.attrs.get("qc_applied") is True
                assert "qc_flagged_counts" in ds.attrs

    def test_validate_output_passes_for_all(self, phase2_pipeline_results: dict[str, list[xr.Dataset]]) -> None:
        stub = _StubForValidation(
            config={},
            model_config={"name": "stub", "type": "nwp", "key": "stub"},
        )
        for model_key, datasets in phase2_pipeline_results.items():
            for ds in datasets:
                assert stub.validate_output(ds) is True
