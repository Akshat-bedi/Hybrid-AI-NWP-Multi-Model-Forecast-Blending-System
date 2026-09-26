"""
src/pipeline/orchestrator.py
----------------------------
Layer   : pipeline
Purpose : Master pipeline orchestrator chaining all operational phases.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import glob
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.blending.base_blender import BlendingError
from src.blending.extreme_booster import ExtremeBooster
from src.blending.inverse_rmse_blender import InverseRMSEBlender
from src.ingestion.base_ingestor import IngestionError
from src.ingestion.demo_ingestor import DemoIngestor
from src.ingestion.gfs_ingestor import GFSIngestor
from src.preprocessing.quality_control import QualityControl
from src.preprocessing.regridder import Regridder
from src.preprocessing.variable_mapper import VariableMapper
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


class ForecastOrchestrator:
    """Master controller executing Phase 1 through Phase 4 sequence."""

    def __init__(self, config: dict, model_config: dict, regions: dict) -> None:
        """Initialize the pipeline orchestrator."""
        self.config = config
        self.model_config = model_config
        self.regions = regions
        
        # Pre-build geographical region masks
        self.region_masks = self._build_region_masks()
        
    def _build_region_masks(self) -> dict[str, np.ndarray]:
        """Convert regions.yaml bounding boxes into 2D boolean masks."""
        lat = np.arange(6.0, 38.25, 0.25)
        lon = np.arange(68.0, 97.25, 0.25)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        
        masks = {"all_india": np.ones_like(lat_grid, dtype=bool)}
        
        if "regions" in self.regions:
            for r_name, r_bounds in self.regions["regions"].items():
                m = (
                    (lat_grid >= r_bounds["lat_min"]) & (lat_grid <= r_bounds["lat_max"]) &
                    (lon_grid >= r_bounds["lon_min"]) & (lon_grid <= r_bounds["lon_max"])
                )
                masks[r_name] = m
                
        return masks
        
    def _get_season(self, month: int) -> str:
        """Map calendar month to meteorological season."""
        for s_name, m_list in {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}.items():
            if month in m_list:
                return s_name
        return "Unknown"

    def run(self, init_time: pd.Timestamp) -> xr.Dataset:
        """Execute the end-to-end forecast blending pipeline for a given init time."""
        start = time.time()
        logger.info("=== Pipeline run: %s ===", init_time)

        # -------------------------------------------------------------
        # Step 1: Ingest all enabled models
        # -------------------------------------------------------------
        model_datasets = {}
        for m_key, m_cfg in self.model_config.get("models", {}).items():
            if not m_cfg.get("enabled", True):
                logger.info("Skipping model %s (disabled)", m_key)
                continue
                
            # Inject key into config for downstream ingestors
            m_cfg["key"] = m_key

            logger.info("Ingesting model: %s", m_key)
            try:
                # Dispatch to appropriate ingestor based on source
                if m_cfg.get("source") == "gfs":
                    ingestor = GFSIngestor(self.config, m_cfg)
                else:
                    ingestor = DemoIngestor(self.config, m_cfg)
                    
                raw_ds = ingestor.ingest(init_time)
                model_datasets[m_key] = raw_ds
                
            except IngestionError as e:
                logger.error("Ingestion failed for %s: %s", m_key, e)
                continue

        if len(model_datasets) < 2:
            raise BlendingError(f"Need at least 2 models to blend. Loaded {len(model_datasets)}.")

        # -------------------------------------------------------------
        # Step 2: Regrid + QC all loaded models
        # -------------------------------------------------------------
        mapper = VariableMapper()
        regridder = Regridder()
        qc = QualityControl()
        
        processed_datasets = {}
        for m_key, ds in model_datasets.items():
            logger.info("Preprocessing model: %s", m_key)
            ds = mapper.map(ds, m_key, self.model_config)
            ds = regridder.regrid(ds)
            ds = qc.check(ds)
            processed_datasets[m_key] = ds

        # -------------------------------------------------------------
        # Step 3: Load latest weight table
        # -------------------------------------------------------------
        skill_dir = Path(self.config["paths"]["skill_scores"])
        parquet_files = glob.glob(str(skill_dir / "*.parquet"))
        
        if parquet_files:
            latest_parquet = max(parquet_files)
            logger.info("Loading latest skill weights from %s", latest_parquet)
            weights_df = pd.read_parquet(latest_parquet)
        else:
            logger.warning("No skill weight parquet found! Blenders will fallback to equal weighting.")
            weights_df = pd.DataFrame(columns=["model", "variable", "region", "lead_hours", "season", "weight"])

        # -------------------------------------------------------------
        # Step 4: Determine season and regime
        # -------------------------------------------------------------
        season = self._get_season(init_time.month)
        regime_id = 0  # Default since classifier isn't actively injecting state here yet

        # -------------------------------------------------------------
        # Step 5: Blend
        # -------------------------------------------------------------
        logger.info("Blending models...")
        meta = {"init_time": init_time, "season": season, "regime_id": regime_id}
        blend_config = load_config(str(Path(self.config["paths"]["raw_data"]).parents[1] / "config" / "blend_config.yaml"))
        
        blender = InverseRMSEBlender(self.config)
        blended = blender.blend(processed_datasets, weights_df, self.region_masks, meta)

        # -------------------------------------------------------------
        # Step 6: Extreme boost
        # -------------------------------------------------------------
        logger.info("Applying extreme weather boosters...")
        booster = ExtremeBooster(blend_config)
        blended = booster.apply(blended, processed_datasets, weights_df)

        # -------------------------------------------------------------
        # Step 7: Save
        # -------------------------------------------------------------
        out_dir = Path(self.config["paths"]["blended_data"])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = out_dir / f"{init_time:%Y%m%dT%H}_blended.nc"
        blended.to_netcdf(out_path)
        
        logger.info("Saved to %s in %.1fs", out_path, time.time() - start)

        return blended
