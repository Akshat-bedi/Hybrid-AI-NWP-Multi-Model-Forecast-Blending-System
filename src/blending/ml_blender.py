"""
src/blending/ml_blender.py
--------------------------
Layer   : blending
Purpose : Machine learning based forecast blending utilizing XGBoost.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.blending.base_blender import BaseBlender
from src.blending.inverse_rmse_blender import InverseRMSEBlender

logger = logging.getLogger(__name__)


class MLBlender(BaseBlender):
    """Blender that trains and evaluates XGBoost regression models per variable."""

    FEATURES = [
        "F_gfs", "F_ecmwf", "F_pangu", "F_graphcast",
        "lead_hours", "sin_doy", "cos_doy", "lat", "lon",
        "regime_id", "season_id"
    ]
    
    SEASON_IDS = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}

    def __init__(self, config: dict, model_dir: str | None = None) -> None:
        """Initialize ML blender and attempt to load any pre-trained models.
        
        Parameters
        ----------
        config : dict
            Pipeline configurations.
        model_dir : str, optional
            Path to the directory containing joblib serialized XGBoost models.
        """
        super().__init__(config)
        self.model_dir = Path(model_dir) if model_dir else None
        self.models: dict[str, xgb.XGBRegressor] = {}

        if self.model_dir and self.model_dir.exists():
            for var in ["t2m", "tp", "u10", "v10", "mslp"]:
                p = self.model_dir / f"ml_blender_{var}.joblib"
                if p.exists():
                    self.models[var] = joblib.load(p)
                    logger.info("Loaded XGBoost model for %s", var)

    def train(
        self, 
        forecast_archive: dict[str, list[xr.Dataset]], 
        truth_archive: list[xr.Dataset], 
        variable: str
    ) -> None:
        """Train an XGBoost blending model for a specific variable.
        
        Flattens spatial dimensions into a massive tabular feature set 
        across time, extracts trigonometric date features, and evaluates
        against a 20% holdout validation set.
        """
        logger.info("Building feature arrays for ML training (%s)...", variable)
        
        X_list = []
        y_list = []
        
        # Determine the number of datasets to process
        # (Assumes all models have aligned datasets with truth_archive)
        n_samples = len(truth_archive)
        
        for i in range(n_samples):
            ds_truth = truth_archive[i]
            if variable not in ds_truth.data_vars:
                continue
                
            # Extract basic coordinates
            lead_hours = ds_truth.coords["lead_hours"].values
            lats = ds_truth.coords["lat"].values
            lons = ds_truth.coords["lon"].values
            
            # Identify time
            ts = pd.Timestamp("2024-01-01") # Default fallback
            for time_coord in ["time", "valid_time"]:
                if time_coord in ds_truth.coords:
                    ts = pd.Timestamp(ds_truth.coords[time_coord].values)
                    break
                    
            doy = ts.dayofyear
            sin_doy = np.sin(2 * np.pi * doy / 365.25)
            cos_doy = np.cos(2 * np.pi * doy / 365.25)
            
            # Simple regime logic (defaulting to 0 since we don't have the Z500 classifier here)
            regime_id = 0 
            
            # Season ID
            month = ts.month
            season_str = "Unknown"
            for s_name, m_list in {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}.items():
                if month in m_list:
                    season_str = s_name
                    break
            season_id = self.SEASON_IDS.get(season_str, 0)

            # Flatten truth
            truth_flat = ds_truth[variable].values.flatten()
            
            # We need to construct X. Shape: (N_gridpoints * N_leads, len(FEATURES))
            n_rows = len(truth_flat)
            X_batch = np.zeros((n_rows, len(self.FEATURES)), dtype=np.float32)
            
            # 1. Forecast models: "F_gfs", "F_ecmwf", "F_pangu", "F_graphcast"
            for j, m_name in enumerate(["gfs", "ecmwf", "pangu", "graphcast"]):
                if m_name in forecast_archive and i < len(forecast_archive[m_name]):
                    ds_m = forecast_archive[m_name][i]
                    if variable in ds_m.data_vars:
                        X_batch[:, j] = ds_m[variable].values.flatten()
                    else:
                        X_batch[:, j] = np.nan
                else:
                    X_batch[:, j] = np.nan
                    
            # 2. lead_hours, lat, lon (Meshgrid broadcasting)
            lh_grid, lat_grid, lon_grid = np.meshgrid(lead_hours, lats, lons, indexing="ij")
            X_batch[:, 4] = lh_grid.flatten()
            
            # 3. Time variables
            X_batch[:, 5] = sin_doy
            X_batch[:, 6] = cos_doy
            
            # 4. Lat / Lon
            X_batch[:, 7] = lat_grid.flatten()
            X_batch[:, 8] = lon_grid.flatten()
            
            # 5. Regime / Season
            X_batch[:, 9] = regime_id
            X_batch[:, 10] = season_id
            
            X_list.append(X_batch)
            y_list.append(truth_flat)

        if not X_list:
            logger.warning("No valid data found to train ML blender for %s", variable)
            return
            
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        # Filter rows where the target (y) is NaN
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            logger.warning("All data for %s was NaN, skipping training.", variable)
            return
            
        # Split 80/20
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Training XGBoost Regressor on %d rows...", len(X_train))
        model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_rmse = root_mean_squared_error(y_train, model.predict(X_train))
        val_rmse = root_mean_squared_error(y_val, model.predict(X_val))
        logger.info("Trained %s ML Blender | Train RMSE: %.4f | Val RMSE: %.4f", variable, train_rmse, val_rmse)
        
        self.models[variable] = model
        
        if self.model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.model_dir / f"ml_blender_{variable}.joblib"
            joblib.dump(model, out_path)
            logger.info("Saved ML blender model to %s", out_path)

    def blend(
        self,
        model_datasets: dict[str, xr.Dataset],
        weights: pd.DataFrame,
        region_masks: dict[str, np.ndarray],
        meta: dict
    ) -> xr.Dataset:
        """Blend datasets using XGBoost.
        
        Falls back to InverseRMSEBlender for variables that lack a trained model.
        """
        init_time = meta.get("init_time", pd.Timestamp("2024-01-01"))
        season_str = meta.get("season", "Unknown")
        regime_id = meta.get("regime_id", 0)
        
        doy = init_time.dayofyear
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        season_id = self.SEASON_IDS.get(season_str, 0)
        
        ref_ds = next(iter(model_datasets.values()))
        lead_hours = ref_ds.coords["lead_hours"].values
        lats = ref_ds.coords["lat"].values
        lons = ref_ds.coords["lon"].values
        
        lh_grid, lat_grid, lon_grid = np.meshgrid(lead_hours, lats, lons, indexing="ij")
        flat_shape = (len(lead_hours) * len(lats) * len(lons),)
        
        # Instantiating the fallback baseline blender
        fallback_blender = InverseRMSEBlender(self.config)
        
        blended_data_vars = {}
        
        for var in ["t2m", "tp", "u10", "v10", "mslp"]:
            if var in self.models:
                # We have a trained XGBoost model for this variable
                X_pred = np.zeros((flat_shape[0], len(self.FEATURES)), dtype=np.float32)
                
                # Fetch F_gfs, F_ecmwf, F_pangu, F_graphcast
                for j, m_name in enumerate(["gfs", "ecmwf", "pangu", "graphcast"]):
                    if m_name in model_datasets and var in model_datasets[m_name]:
                        X_pred[:, j] = model_datasets[m_name][var].values.flatten()
                    else:
                        X_pred[:, j] = np.nan
                        
                X_pred[:, 4] = lh_grid.flatten()
                X_pred[:, 5] = sin_doy
                X_pred[:, 6] = cos_doy
                X_pred[:, 7] = lat_grid.flatten()
                X_pred[:, 8] = lon_grid.flatten()
                X_pred[:, 9] = regime_id
                X_pred[:, 10] = season_id
                
                # XGBoost can natively handle NaNs during inference (treats as missing split)
                preds = self.models[var].predict(X_pred)
                
                blended_var = preds.reshape((len(lead_hours), len(lats), len(lons)))
                # Optional: boundary smoothing? The ML model might not have hard boundaries
                # but we'll apply it for consistency across methods if needed.
                blended_var = self._apply_boundary_smoothing(blended_var, sigma=1.0)
                
                blended_data_vars[var] = xr.DataArray(blended_var, dims=["lead_hours", "lat", "lon"])
                
            else:
                # Fallback to InverseRMSE for this variable
                logger.debug("No ML model for %s, utilizing InverseRMSEBlender fallback.", var)
                fallback_ds = fallback_blender.blend(model_datasets, weights, region_masks, meta)
                blended_data_vars[var] = fallback_ds[var]
                
        # Reconstruct final dataset
        blended_ds = xr.Dataset(
            data_vars=blended_data_vars,
            coords={"lead_hours": lead_hours, "lat": lats, "lon": lons},
            attrs={
                "model_name": "hybrid_blend",
                "model_type": "ensemble",
                "blend_method": "ml_xgboost",
                "units": ref_ds.attrs.get("units", "{}")
            }
        )
        
        return blended_ds
