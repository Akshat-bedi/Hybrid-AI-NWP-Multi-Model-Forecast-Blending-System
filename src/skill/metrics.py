"""
src/skill/metrics.py
--------------------
Layer   : skill
Purpose : Compute verification metrics (RMSE, MAE, Bias, ETS, Spearman).

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import numpy as np
import scipy.stats


def rmse(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Compute Root Mean Square Error (RMSE), ignoring NaNs."""
    return float(np.sqrt(np.nanmean((forecast - truth) ** 2)))


def mae(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE), ignoring NaNs."""
    return float(np.nanmean(np.abs(forecast - truth)))


def bias(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Compute systematic bias (mean error), ignoring NaNs."""
    return float(np.nanmean(forecast - truth))


def ets(forecast: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    """Compute Equitable Threat Score (ETS) for categorical events.

    Parameters
    ----------
    forecast : np.ndarray
        Forecast values.
    truth : np.ndarray
        Observed/truth values.
    threshold : float
        Threshold for binarizing the events.

    Returns
    -------
    float
        The ETS score. Returns 0.0 if the denominator is zero.
    """
    valid_mask = ~np.isnan(forecast) & ~np.isnan(truth)
    f_valid = forecast[valid_mask]
    t_valid = truth[valid_mask]

    f_bin = f_valid >= threshold
    t_bin = t_valid >= threshold

    hits = np.sum(f_bin & t_bin)
    misses = np.sum(~f_bin & t_bin)
    false_alarms = np.sum(f_bin & ~t_bin)
    correct_negatives = np.sum(~f_bin & ~t_bin)

    total = hits + misses + false_alarms + correct_negatives
    if total == 0:
        return 0.0

    hits_random = (hits + misses) * (hits + false_alarms) / total
    denominator = hits + misses + false_alarms - hits_random

    if denominator == 0:
        return 0.0

    return float((hits - hits_random) / denominator)


def spearman_corr(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Compute Spearman rank correlation, ignoring NaNs."""
    valid_mask = ~np.isnan(forecast) & ~np.isnan(truth)
    if not np.any(valid_mask):
        return np.nan

    f_valid = forecast[valid_mask]
    t_valid = truth[valid_mask]

    corr, _ = scipy.stats.spearmanr(f_valid, t_valid)
    return float(corr) if not np.isnan(corr) else 0.0


def skill_score(metric_value: float, reference_value: float) -> float:
    """Compute generic skill score.

    Score = 1 - (metric_value / reference_value)

    Returns 0.0 if reference_value is zero.
    """
    if reference_value == 0:
        return 0.0
    return float(1.0 - (metric_value / reference_value))
