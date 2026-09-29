"""
tests/test_phase1_integration.py
---------------------------------
CHECK 3 — Phase 0 → Phase 1 end-to-end integration test.

Full pipeline sequence exercised:
  1. load_config()               [Phase 0 — config layer]
  2. generate_demo_data.generate()  [Phase 0 — data generation]
  3. DemoIngestor.ingest()       [Phase 1 — ingestion layer]
  4. BaseIngestor.validate_output() [Phase 1 — contract validation]

The test:
  - Generates 2 days of demo data into an isolated tmp directory
  - Ingests all 4 demo models (pangu, graphcast, ecmwf, gfs) for both days
  - Asserts every returned xr.Dataset is contract-compliant
  - Asserts no unhandled exceptions are raised at any stage

No real network calls or external files are required.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging
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

from src.ingestion.base_ingestor import BaseIngestor  # noqa: E402
from src.ingestion.demo_ingestor import DemoIngestor  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

_START = date(2024, 6, 1)
_END = date(2024, 6, 2)
_N_DAYS = (_END - _START).days + 1

# All four demo models with their YAML keys and expected model_type
_DEMO_MODELS: dict[str, str] = {
    "pangu": "ai",
    "graphcast": "ai",
    "ecmwf": "nwp",
    "gfs": "nwp",
}

_REQUIRED_VARS: list[str] = ["t2m", "tp", "u10", "v10", "mslp"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_cfg(raw_root: Path) -> dict:
    """Return a pipeline config dict pointing at *raw_root*."""
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


def _iter_dates() -> list[date]:
    """Return dates in [_START, _END] inclusive."""
    return [_START + timedelta(days=i) for i in range(_N_DAYS)]


# ---------------------------------------------------------------------------
# Class-scoped fixture — Phase 0 data generation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def raw_root_with_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run generate() into a tmp directory and return the raw_root Path.

    Uses ``unittest.mock.patch.object`` (scope-agnostic context manager)
    to redirect pipeline paths without requiring a function-scoped
    ``monkeypatch`` fixture.
    """
    import scripts.generate_demo_data as gdd

    raw_root = tmp_path_factory.mktemp("phase1_raw")
    real_blend_cfg = load_config(
        str(PROJECT_ROOT / "config" / "blend_config.yaml")
    )
    tmp_pipeline_cfg = _make_pipeline_cfg(raw_root)

    def _mock_load_config(path: str) -> dict:
        return real_blend_cfg if "blend_config" in path else tmp_pipeline_cfg

    with patch.object(gdd, "load_config", side_effect=_mock_load_config):
        gdd.generate(start=_START, end=_END)

    return raw_root


# ===========================================================================
# CHECK 3 — Phase 0: config layer integrity
# ===========================================================================

class TestPhase0Config:
    """Phase 0 config layer must load cleanly before Phase 1 can run."""

    def test_blend_config_loads(self) -> None:
        cfg = load_config(str(PROJECT_ROOT / "config" / "blend_config.yaml"))
        assert isinstance(cfg, dict)
        assert "blending_method" in cfg
        assert "demo_models" in cfg

    def test_pipeline_config_loads(self) -> None:
        cfg = load_config(str(PROJECT_ROOT / "config" / "pipeline_config.yaml"))
        assert isinstance(cfg, dict)
        assert "paths" in cfg

    def test_models_config_loads(self) -> None:
        cfg = load_config(str(PROJECT_ROOT / "config" / "models.yaml"))
        assert set(cfg["models"].keys()) == {"gfs", "ecmwf", "pangu", "graphcast", "open_meteo"}

    def test_regions_config_loads(self) -> None:
        cfg = load_config(str(PROJECT_ROOT / "config" / "regions.yaml"))
        assert len(cfg["regions"]) == 6


# ===========================================================================
# CHECK 3 — Phase 0: generated files exist on disk
# ===========================================================================

class TestPhase0GeneratedFiles:
    """Phase 0 output: all expected NetCDF files must be present."""

    def test_all_model_files_exist(self, raw_root_with_data: Path) -> None:
        for day in _iter_dates():
            for model_key in _DEMO_MODELS:
                path = raw_root_with_data / model_key / f"{day.strftime('%Y%m%d')}.nc"
                assert path.exists(), f"Missing: {path}"

    def test_all_truth_files_exist(self, raw_root_with_data: Path) -> None:
        for day in _iter_dates():
            path = (
                raw_root_with_data
                / "observations"
                / f"truth_{day.strftime('%Y%m%d')}.nc"
            )
            assert path.exists(), f"Missing truth: {path}"

    def test_file_count_correct(self, raw_root_with_data: Path) -> None:
        expected = _N_DAYS * len(_DEMO_MODELS)
        actual = sum(
            1
            for model_key in _DEMO_MODELS
            for _ in (raw_root_with_data / model_key).glob("*.nc")
        )
        assert actual == expected


# ===========================================================================
# CHECK 3 — Phase 1: DemoIngestor reads all generated files
# ===========================================================================

class TestPhase1DemoIngestorE2E:
    """Phase 1: DemoIngestor must successfully ingest every generated file."""

    @pytest.fixture(scope="class")
    @classmethod
    def ingested_results(
        cls, raw_root_with_data: Path
    ) -> dict[str, list[xr.Dataset]]:
        """Ingest all models for all dates; return {model_key: [ds, ...]}."""
        pipeline_cfg = {"paths": {"raw_data": str(raw_root_with_data)}}
        results: dict[str, list[xr.Dataset]] = {k: [] for k in _DEMO_MODELS}

        for model_key, model_type in _DEMO_MODELS.items():
            model_config = {
                "name": model_key,
                "type": model_type,
                "source": "demo",
                "key": model_key,
            }
            ingestor = DemoIngestor(pipeline_cfg, model_config)
            for day in _iter_dates():
                ts = pd.Timestamp(day.strftime("%Y-%m-%d"))
                ds = ingestor.ingest(ts)
                results[model_key].append(ds)

        return results

    # --- Core pipeline assertions ------------------------------------------

    def test_no_unhandled_exceptions(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """If ingested_results was built without exception, this always passes."""
        assert ingested_results is not None

    def test_all_models_ingested(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        for model_key in _DEMO_MODELS:
            assert model_key in ingested_results
            assert len(ingested_results[model_key]) == _N_DAYS

    # --- CHECK 1 assertions applied to Phase 1 output ---------------------

    def test_all_results_are_xr_dataset(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """Assertion 4: isinstance(result, xr.Dataset) for every ingested file."""
        for model_key, datasets in ingested_results.items():
            for ds in datasets:
                assert isinstance(ds, xr.Dataset), (
                    f"{model_key}: ingest() returned {type(ds).__name__}"
                )

    def test_all_results_have_correct_dims(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """Assertion 5: set(result.dims) == {"lead_hours", "lat", "lon"}."""
        for model_key, datasets in ingested_results.items():
            for ds in datasets:
                assert set(ds.dims) == {"lead_hours", "lat", "lon"}, (
                    f"{model_key}: wrong dims {set(ds.dims)}"
                )

    def test_all_results_have_required_variables(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """Assertion 6: all 5 contract variables present in every result."""
        for model_key, datasets in ingested_results.items():
            for ds in datasets:
                missing = [v for v in _REQUIRED_VARS if v not in ds]
                assert not missing, (
                    f"{model_key}: missing variables {missing}"
                )

    def test_all_results_have_model_name_attr(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """Assertion 7: result.attrs.get("model_name") is not None."""
        for model_key, datasets in ingested_results.items():
            for ds in datasets:
                assert ds.attrs.get("model_name") is not None, (
                    f"{model_key}: model_name attr is None"
                )

    def test_validate_output_passes_for_all(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """Assertion 3: BaseIngestor.validate_output() returns True for each."""
        stub = _StubForValidation(
            config={"paths": {"raw_data": "data/raw"}},
            model_config={"name": "stub", "type": "nwp", "key": "stub"},
        )
        for model_key, datasets in ingested_results.items():
            for ds in datasets:
                assert stub.validate_output(ds) is True, (
                    f"Contract validation failed for model '{model_key}'"
                )

    # --- Phase 1-specific correctness checks --------------------------------

    def test_model_type_attrs_correct(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """model_type attribute must match models.yaml: nwp or ai."""
        for model_key, datasets in ingested_results.items():
            expected_type = _DEMO_MODELS[model_key]
            for ds in datasets:
                assert ds.attrs.get("model_type") == expected_type, (
                    f"{model_key}: expected model_type='{expected_type}', "
                    f"got '{ds.attrs.get('model_type')}'"
                )

    def test_tp_non_negative_in_all_results(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """tp must be >= 0 everywhere (clipped during generation)."""
        for model_key, datasets in ingested_results.items():
            for ds in datasets:
                assert float(ds["tp"].min()) >= 0.0, (
                    f"{model_key}: negative tp detected"
                )

    def test_lead_hours_exact_values(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """lead_hours coordinate must be exactly [0, 24, 48, 72, 96, 120]."""
        expected = [0, 24, 48, 72, 96, 120]
        for model_key, datasets in ingested_results.items():
            for ds in datasets:
                np.testing.assert_array_equal(
                    ds.coords["lead_hours"].values,
                    expected,
                    err_msg=f"{model_key}: lead_hours mismatch",
                )

    def test_rmse_ordering_preserved_after_ingest(
        self, ingested_results: dict[str, list[xr.Dataset]]
    ) -> None:
        """Phase 0 noise guarantee: pangu spread < graphcast spread for t2m.

        Compares the standard deviation of t2m across all ingested pangu
        and graphcast files.  Because graphcast was generated with a higher
        noise_std (2.5 vs 0.9), its temporal variability must be larger.
        This confirms Phase 1 faithfully preserves Phase 0 output without
        data corruption or accidental smoothing.
        """
        pangu_vals = np.concatenate(
            [ds["t2m"].values.ravel() for ds in ingested_results["pangu"]]
        )
        graphcast_vals = np.concatenate(
            [ds["t2m"].values.ravel() for ds in ingested_results["graphcast"]]
        )
        pangu_std = float(np.std(pangu_vals))
        graphcast_std = float(np.std(graphcast_vals))
        logger.info(
            "t2m std — pangu: %.4f, graphcast: %.4f", pangu_std, graphcast_std
        )
        assert pangu_std < graphcast_std, (
            f"pangu std ({pangu_std:.4f}) should be < "
            f"graphcast std ({graphcast_std:.4f}) — "
            "noise ordering not preserved through ingest"
        )


# ===========================================================================
# Helpers
# ===========================================================================

class _StubForValidation(BaseIngestor):
    """Minimal concrete ingestor used only to call validate_output()."""

    def ingest(self, date: pd.Timestamp, lead_hours: list[int]) -> xr.Dataset:  # noqa: D102
        raise NotImplementedError("validation stub only")
