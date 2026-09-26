"""
src/blending/base_blender.py
----------------------------
Layer   : blending
Purpose : Abstract base class for the model blending logic.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter


class BlendingError(Exception):
    """Exception raised for errors during blending."""
    pass


class BaseBlender(ABC):
    """Abstract base class for all blending algorithms."""

    def __init__(self, config: dict) -> None:
        """Initialize the blender with pipeline configurations."""
        self.config = config

    @abstractmethod
    def blend(
        self,
        model_datasets: dict[str, xr.Dataset],
        weights: pd.DataFrame,
        region_masks: dict[str, np.ndarray],
        meta: dict
    ) -> xr.Dataset:
        """Blend the input model datasets using the specified weights.

        Parameters
        ----------
        model_datasets : dict[str, xr.Dataset]
            Dictionary mapping model names to their respective forecasted xr.Dataset.
        weights : pd.DataFrame
            DataFrame containing weights for models per region and variable.
        region_masks : dict[str, np.ndarray]
            Dictionary mapping region keys to 2D boolean numpy arrays.
        meta : dict
            Metadata dictionary containing:
              - "init_time": pd.Timestamp
              - "season": str
              - "regime_id": int

        Returns
        -------
        xr.Dataset
            A blended dataset matching the data contract standard with 
            an additional attribute 'blend_method' set to the blender identifier.
        """
        pass

    def _apply_boundary_smoothing(self, arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Smooth boundaries between regions in the blended field.
        
        Applies scipy.ndimage.gaussian_filter on 2D spatial fields to reduce 
        hard seams generated when splicing grid regions with varying weights.

        Parameters
        ----------
        arr : np.ndarray
            The array to smooth. Expects 2D (lat, lon) or 3D (lead_hours, lat, lon).
        sigma : float, optional
            Gaussian kernel standard deviation, by default 1.0.

        Returns
        -------
        np.ndarray
            The smoothed numpy array.
        """
        if arr.ndim == 2:
            return gaussian_filter(arr, sigma=sigma)
        elif arr.ndim == 3:
            smoothed = np.empty_like(arr)
            # Apply 2D spatial smoothing independently to each lead_hour
            for i in range(arr.shape[0]):
                smoothed[i] = gaussian_filter(arr[i], sigma=sigma)
            return smoothed
        else:
            raise ValueError(f"Expected 2D or 3D array for smoothing, got {arr.ndim}D")
