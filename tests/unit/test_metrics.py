"""
tests/unit/test_metrics.py
--------------------------
Unit tests for the skill evaluation metrics.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pytest

from src.skill import metrics


def test_rmse() -> None:
    forecast = np.array([1.0, 2.0, 3.0, np.nan])
    truth = np.array([1.0, 4.0, 3.0, 5.0])
    
    # Diff: 0.0, -2.0, 0.0. Squared: 0.0, 4.0, 0.0. Mean sq: 4.0/3.
    # RMSE: sqrt(4/3) = 1.1547...
    expected = np.sqrt(4.0 / 3.0)
    assert np.isclose(metrics.rmse(forecast, truth), expected)


def test_mae() -> None:
    forecast = np.array([1.0, 2.0, 3.0, np.nan])
    truth = np.array([1.0, 4.0, 3.0, 5.0])
    # Abs diff: 0.0, 2.0, 0.0. Mean: 2.0/3.
    expected = 2.0 / 3.0
    assert np.isclose(metrics.mae(forecast, truth), expected)


def test_bias() -> None:
    forecast = np.array([1.0, 2.0, 3.0, np.nan])
    truth = np.array([2.0, 4.0, 3.0, 5.0])
    # Diff: -1.0, -2.0, 0.0. Mean: -3.0/3 = -1.0
    expected = -1.0
    assert np.isclose(metrics.bias(forecast, truth), expected)


def test_ets_known_values() -> None:
    """Test ETS explicitly with known hits, misses, false_alarms."""
    # Threshold = 1.0
    forecast = np.array([1.0, 1.0, 0.0, 1.0, 0.0, np.nan])
    truth =    np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    
    # valid points: indices 0, 1, 2, 3, 4 (total 5 points)
    # f_bin: [True, True, False, True, False]
    # t_bin: [True, True, True, False, False]
    
    # Hits: (0, 1) -> 2
    # Misses: (2) -> 1
    # False Alarms: (3) -> 1
    # Correct Negatives: (4) -> 1
    
    # total = 5
    # hits_random = (hits + misses) * (hits + FA) / total = (3 * 3) / 5 = 1.8
    # denominator = hits + misses + FA - hits_random = 2 + 1 + 1 - 1.8 = 2.2
    # ETS = (hits - hits_random) / denominator = (2 - 1.8) / 2.2 = 0.2 / 2.2 = 1 / 11
    
    val = metrics.ets(forecast, truth, threshold=1.0)
    assert np.isclose(val, 1.0 / 11.0)


def test_ets_zero_denominator() -> None:
    """Test ETS returns 0.0 when denominator is 0."""
    forecast = np.array([0.0, 0.0, 0.0])
    truth = np.array([0.0, 0.0, 0.0])
    assert metrics.ets(forecast, truth, threshold=1.0) == 0.0


def test_spearman_corr() -> None:
    forecast = np.array([1.0, 2.0, 3.0, np.nan])
    truth = np.array([1.0, 2.0, 3.0, 5.0])
    
    corr = metrics.spearman_corr(forecast, truth)
    assert np.isclose(corr, 1.0)


def test_spearman_corr_all_nan() -> None:
    forecast = np.array([np.nan, np.nan])
    truth = np.array([1.0, 2.0])
    
    corr = metrics.spearman_corr(forecast, truth)
    assert np.isnan(corr)


def test_skill_score() -> None:
    assert metrics.skill_score(1.0, 2.0) == 0.5
    assert metrics.skill_score(1.0, 0.0) == 0.0
