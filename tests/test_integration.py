"""
tests/test_integration.py
-------------------------
CHECK 3 — Phase-0 end-to-end integration test.

Exercises the complete Phase-0 pipeline in sequence:
  1. load_config() correctly parses all 4 YAML config files
  2. generate() writes conformant NetCDF files for every model + truth
  3. Every written file loads back as a contract-compliant xr.Dataset
  4. BaseIngestor.validate_output() passes for all files
  5. Noise-std ordering guarantees the expected RMSE ranking

The test uses a 2-day window and redirects all output to a pytest
``tmp_path``-isolated directory so it leaves no artifacts in the tree.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.base import BaseIngestor  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test-scope constants
# ---------------------------------------------------------------------------
_START = date(2024, 6, 1)
_END = date(2024, 6, 2)          # 2 days → fast, still exercises the day loop
_MODELS: tuple[str, ...] = ("gfs", "ecmwf", "pangu", "graphcast")
_N_DAYS = (_END - _START).days + 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_pipeline_cfg(raw_root: Path) -> dict:
    """Return a pipeline config dict with all paths pointing to *raw_root*."""
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


def _iter_days() -> list[date]:
    """Return the list of dates in [_START, _END]."""
    return [_START + timedelta(days=i) for i in range(_N_DAYS)]


# ---------------------------------------------------------------------------
# Fixture — run the generator once per class, isolated to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def generated_raw_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run generate() into a tmp directory and return the raw_root Path.

    ``load_config`` inside the generate module is patched via
    ``unittest.mock.patch.object`` (a context manager that is scope-agnostic)
    so that:
    - blend_config.yaml → real config (needed for noise params)
    - pipeline_config.yaml → synthetic config pointing at tmp_path

    No ``monkeypatch`` fixture is used; this keeps the fixture compatible
    with ``scope="class"``.
    """
    from unittest.mock import patch
    import scripts.generate_demo_data as gdd

    raw_root = tmp_path_factory.mktemp("raw")
    real_blend_cfg = load_config(str(PROJECT_ROOT / "config" / "blend_config.yaml"))
    tmp_pipeline = _tmp_pipeline_cfg(raw_root)

    def _mock_load_config(path: str) -> dict:
        return real_blend_cfg if "blend_config" in path else tmp_pipeline

    with patch.object(gdd, "load_config", side_effect=_mock_load_config):
        gdd.generate(start=_START, end=_END)

    return raw_root


# ===========================================================================
# CHECK 3 — config layer
# ===========================================================================

class TestConfigLayer:
    """load_config() must parse every YAML and behave correctly on errors."""

    def test_all_configs_return_dict(self) -> None:
        """Each YAML config file must deserialise to a plain dict."""
        for name in ("models.yaml", "regions.yaml", "blend_config.yaml", "pipeline_config.yaml"):
            cfg = load_config(str(PROJECT_ROOT / "config" / name))
            assert isinstance(cfg, dict), f"load_config returned non-dict for {name}"

    def test_models_yaml_has_five_entries(self) -> None:
        """models.yaml must register exactly 5 models."""
        cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
        assert len(cfg["models"]) == 5

    def test_regions_yaml_has_six_entries(self) -> None:
        """regions.yaml must define exactly 6 subregions."""
        cfg = load_config(str(PROJECT_ROOT / "config" / "regions.yaml"))
        assert len(cfg["regions"]) == 6

    def test_blend_config_has_required_keys(self) -> None:
        """blend_config.yaml must contain blending_method, thresholds, demo_models."""
        cfg = load_config(str(PROJECT_ROOT / "config" / "blend_config.yaml"))
        for key in ("blending_method", "thresholds", "demo_models"):
            assert key in cfg, f"Missing key '{key}' in blend_config.yaml"

    def test_pipeline_config_has_required_keys(self) -> None:
        """pipeline_config.yaml must contain paths, scheduler, api."""
        cfg = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
        for key in ("paths", "scheduler", "api"):
            assert key in cfg, f"Missing key '{key}' in pipeline_config.yaml"

    def test_load_config_raises_on_missing_file(self) -> None:
        """FileNotFoundError with a descriptive message for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("this/does/not/exist.yaml")


# ===========================================================================
# CHECK 3 — file existence
# ===========================================================================

class TestFileExistence:
    """generate() must produce exactly the expected set of output files."""

    def test_model_files_exist(self, generated_raw_root: Path) -> None:
        """One NetCDF per model per day must be present."""
        for day in _iter_days():
            for model in _MODELS:
                path = generated_raw_root / model / f"{day.strftime('%Y%m%d')}.nc"
                assert path.exists(), f"Expected file missing: {path}"

    def test_truth_files_exist(self, generated_raw_root: Path) -> None:
        """One truth NetCDF per day must be present."""
        for day in _iter_days():
            path = generated_raw_root / "observations" / f"truth_{day.strftime('%Y%m%d')}.nc"
            assert path.exists(), f"Expected truth file missing: {path}"

    def test_total_model_file_count(self, generated_raw_root: Path) -> None:
        """Total model files must equal n_days × n_models."""
        expected = _N_DAYS * len(_MODELS)
        actual = sum(
            1 for model in _MODELS
            for _ in (generated_raw_root / model).glob("*.nc")
        )
        assert actual == expected, f"Expected {expected} model files, found {actual}"


# ===========================================================================
# CHECK 3 — data contract on loaded files
# ===========================================================================

class TestLoadedFilesPassDataContract:
    """Every generated NetCDF must load as a contract-compliant xr.Dataset."""

    def test_model_outputs_pass_validate_output(self, generated_raw_root: Path) -> None:
        """BaseIngestor.validate_output() must pass for every model file."""
        for day in _iter_days():
            for model in _MODELS:
                path = generated_raw_root / model / f"{day.strftime('%Y%m%d')}.nc"
                with xr.open_dataset(path) as ds:
                    assert isinstance(ds, xr.Dataset)
                    assert BaseIngestor.validate_output(ds), (
                        f"Contract validation failed for {path}"
                    )

    def test_truth_files_pass_validate_output(self, generated_raw_root: Path) -> None:
        """BaseIngestor.validate_output() must pass for every truth file."""
        for day in _iter_days():
            path = generated_raw_root / "observations" / f"truth_{day.strftime('%Y%m%d')}.nc"
            with xr.open_dataset(path) as ds:
                assert isinstance(ds, xr.Dataset)
                assert BaseIngestor.validate_output(ds)

    def test_truth_files_have_correct_attrs(self, generated_raw_root: Path) -> None:
        """Truth files must carry model_name='truth' and model_type='ensemble'."""
        for day in _iter_days():
            path = generated_raw_root / "observations" / f"truth_{day.strftime('%Y%m%d')}.nc"
            with xr.open_dataset(path) as ds:
                assert ds.attrs.get("model_name") == "truth"
                assert ds.attrs.get("model_type") == "ensemble"

    def test_model_type_attrs_correct(self, generated_raw_root: Path) -> None:
        """NWP models must carry model_type='nwp'; AI models 'ai'."""
        expected_types = {"gfs": "nwp", "ecmwf": "nwp", "pangu": "ai", "graphcast": "ai"}
        day = _iter_days()[0]
        for model, expected_type in expected_types.items():
            path = generated_raw_root / model / f"{day.strftime('%Y%m%d')}.nc"
            with xr.open_dataset(path) as ds:
                assert ds.attrs.get("model_type") == expected_type, (
                    f"{model}: expected model_type='{expected_type}', "
                    f"got '{ds.attrs.get('model_type')}'"
                )

    def test_tp_non_negative_in_all_model_files(self, generated_raw_root: Path) -> None:
        """tp must be >= 0 in every model file (np.clip applied during generation)."""
        for day in _iter_days():
            for model in _MODELS:
                path = generated_raw_root / model / f"{day.strftime('%Y%m%d')}.nc"
                with xr.open_dataset(path) as ds:
                    assert float(ds["tp"].min()) >= 0.0, (
                        f"Negative tp found in {model} / {day}"
                    )


# ===========================================================================
# CHECK 3 — skill ordering (noise-std contract guarantee)
# ===========================================================================

class TestSkillOrdering:
    """The noise_std hierarchy must produce the expected RMSE ranking."""

    def test_rmse_ordering_pangu_best_graphcast_worst(
        self, generated_raw_root: Path
    ) -> None:
        """pangu RMSE < ecmwf RMSE < gfs RMSE < graphcast RMSE for t2m."""
        rmse: dict[str, list[float]] = {m: [] for m in _MODELS}

        for day in _iter_days():
            truth_path = generated_raw_root / "observations" / f"truth_{day.strftime('%Y%m%d')}.nc"
            truth_arr = xr.open_dataset(truth_path)["t2m"].values
            for model in _MODELS:
                pred_path = generated_raw_root / model / f"{day.strftime('%Y%m%d')}.nc"
                pred_arr = xr.open_dataset(pred_path)["t2m"].values
                rmse[model].append(
                    float(np.sqrt(np.mean((pred_arr - truth_arr) ** 2)))
                )

        mean_rmse = {m: float(np.mean(v)) for m, v in rmse.items()}
        logger.info("Mean t2m RMSE: %s", mean_rmse)

        # Guaranteed by noise_std: 0.9 < 1.2 < 1.5 < 2.5
        assert mean_rmse["pangu"] < mean_rmse["ecmwf"], (
            f"pangu RMSE ({mean_rmse['pangu']:.4f}) should be < "
            f"ecmwf RMSE ({mean_rmse['ecmwf']:.4f})"
        )
        assert mean_rmse["ecmwf"] < mean_rmse["gfs"], (
            f"ecmwf RMSE ({mean_rmse['ecmwf']:.4f}) should be < "
            f"gfs RMSE ({mean_rmse['gfs']:.4f})"
        )
        assert mean_rmse["gfs"] < mean_rmse["graphcast"], (
            f"gfs RMSE ({mean_rmse['gfs']:.4f}) should be < "
            f"graphcast RMSE ({mean_rmse['graphcast']:.4f})"
        )
