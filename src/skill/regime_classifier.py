"""
src/skill/regime_classifier.py
------------------------------
Layer   : skill
Purpose : Classify large-scale circulation regimes using Z500 anomalies.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.cluster import KMeans


class RegimeClassifier:
    """Classifies atmospheric states into discrete regimes based on Z500 anomalies."""

    def __init__(self, n_regimes: int = 4) -> None:
        """Initialize the classifier with the number of regimes."""
        self.n_regimes = n_regimes
        self.kmeans: KMeans | None = None
        self.climatology: np.ndarray | None = None

    def fit(self, z500_data: np.ndarray) -> RegimeClassifier:
        """Fit the KMeans clustering on Z500 data.

        Computes the climatological mean and fits on the anomalies.

        Parameters
        ----------
        z500_data : np.ndarray
            Shape (n_timesteps, n_lat * n_lon), already flattened per timestep.
        
        Returns
        -------
        RegimeClassifier
            The fitted instance.
        """
        if z500_data is None or z500_data.size == 0:
            return self

        self.climatology = np.nanmean(z500_data, axis=0)
        anomalies = z500_data - self.climatology

        # Replace remaining NaNs with 0 (climatology) for KMeans
        anomalies = np.nan_to_num(anomalies, nan=0.0)

        self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        self.kmeans.fit(anomalies)
        
        return self

    def predict(self, z500_data: np.ndarray) -> np.ndarray:
        """Predict the regime labels for the given Z500 data.

        If Z500 data is missing or the model hasn't been fitted, degrades gracefully
        and returns an array of zeros.

        Parameters
        ----------
        z500_data : np.ndarray
            Shape (n_timesteps, n_lat * n_lon).

        Returns
        -------
        np.ndarray
            Array of predicted regime labels (integers 0 to n_regimes-1).
        """
        n_samples = z500_data.shape[0] if (z500_data is not None and len(z500_data.shape) > 0) else 1
        
        if z500_data is None or z500_data.size == 0 or self.kmeans is None or self.climatology is None:
            return np.zeros(n_samples, dtype=int)

        anomalies = z500_data - self.climatology
        anomalies = np.nan_to_num(anomalies, nan=0.0)
        
        return self.kmeans.predict(anomalies)

    def save(self, path: str) -> None:
        """Save the fitted model to disk using joblib.

        Parameters
        ----------
        path : str
            The destination file path.
        """
        joblib.dump(self, path)

    def load(self, path: str) -> RegimeClassifier:
        """Load a fitted model from disk.

        Parameters
        ----------
        path : str
            The path to the saved model file.
            
        Returns
        -------
        RegimeClassifier
            The loaded instance (modifies self in-place and returns self).
        """
        loaded = joblib.load(path)
        self.n_regimes = loaded.n_regimes
        self.kmeans = loaded.kmeans
        self.climatology = loaded.climatology
        return self
