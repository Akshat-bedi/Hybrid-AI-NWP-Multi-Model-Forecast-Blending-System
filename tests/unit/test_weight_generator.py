"""
tests/unit/test_weight_generator.py
-----------------------------------
Unit tests for the weight generator.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
"""

import numpy as np
import pandas as pd
import pytest

from src.skill.weight_generator import compute_weights


def test_compute_weights_inverse_rmse() -> None:
    """Test inverse RMSE weighting method."""
    df = pd.DataFrame({
        "model": ["A", "B"],
        "variable": ["t2m", "t2m"],
        "region": ["all_india", "all_india"],
        "lead_hours": [0, 0],
        "season": ["DJF", "DJF"],
        "rmse": [1.0, 3.0]
    })
    
    out = compute_weights(df, method="inverse_rmse")
    
    # inv_rmse = [1.0, 0.3333] -> sum = 1.3333
    # w_A = 1.0 / 1.3333 = 0.75
    # w_B = 0.3333 / 1.3333 = 0.25
    weights = out["weight"].values
    assert np.isclose(weights[0], 0.75)
    assert np.isclose(weights[1], 0.25)
    assert np.isclose(np.sum(weights), 1.0)


def test_compute_weights_inverse_rmse_perfect_forecast() -> None:
    """Test inverse RMSE handles RMSE = 0 gracefully."""
    df = pd.DataFrame({
        "model": ["A", "B", "C"],
        "variable": ["t2m", "t2m", "t2m"],
        "region": ["all_india", "all_india", "all_india"],
        "lead_hours": [0, 0, 0],
        "season": ["DJF", "DJF", "DJF"],
        "rmse": [0.0, 2.0, 0.0]
    })
    
    out = compute_weights(df, method="inverse_rmse")
    
    # Models A and C have 0 RMSE -> should split weight evenly (0.5). B = 0.
    weights = out["weight"].values
    assert np.isclose(weights[0], 0.5)
    assert np.isclose(weights[1], 0.0)
    assert np.isclose(weights[2], 0.5)
    assert np.isclose(np.sum(weights), 1.0)


def test_compute_weights_rank() -> None:
    """Test rank-based weighting method."""
    df = pd.DataFrame({
        "model": ["A", "B", "C"],
        "variable": ["t2m", "t2m", "t2m"],
        "region": ["all_india", "all_india", "all_india"],
        "lead_hours": [0, 0, 0],
        "season": ["DJF", "DJF", "DJF"],
        "rmse": [10.0, 1.0, 5.0]
    })
    
    out = compute_weights(df, method="rank")
    
    # Ranks: A=3, B=1, C=2
    # inv_rank: A=1/3, B=1, C=1/2 -> sum = 11/6
    # w_A = (1/3) / (11/6) = 2/11
    # w_B = 1 / (11/6) = 6/11
    # w_C = (1/2) / (11/6) = 3/11
    weights = out["weight"].values
    assert np.isclose(weights[0], 2.0 / 11.0)
    assert np.isclose(weights[1], 6.0 / 11.0)
    assert np.isclose(weights[2], 3.0 / 11.0)
    assert np.isclose(np.sum(weights), 1.0)


def test_compute_weights_multiple_groups() -> None:
    """Test that weights are computed independently per group."""
    df = pd.DataFrame({
        "model": ["A", "B", "A", "B"],
        "variable": ["t2m", "t2m", "tp", "tp"],  # Two different variables -> two groups
        "region": ["all_india", "all_india", "all_india", "all_india"],
        "lead_hours": [0, 0, 0, 0],
        "season": ["DJF", "DJF", "DJF", "DJF"],
        "rmse": [1.0, 3.0, 2.0, 2.0]
    })
    
    out = compute_weights(df, method="inverse_rmse")
    
    # Group 1 (t2m): w_A = 0.75, w_B = 0.25
    # Group 2 (tp): w_A = 0.5, w_B = 0.5
    assert np.isclose(out.loc[0, "weight"], 0.75)
    assert np.isclose(out.loc[1, "weight"], 0.25)
    assert np.isclose(out.loc[2, "weight"], 0.5)
    assert np.isclose(out.loc[3, "weight"], 0.5)
