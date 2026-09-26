"""
src/skill/weight_generator.py
-----------------------------
Layer   : skill
Purpose : Generate optimal model weights based on historical skill.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weights(skill_df: pd.DataFrame, method: str = "inverse_rmse") -> pd.DataFrame:
    """Compute optimal model weights based on historical skill metrics.
    
    Groups by [variable, region, lead_hours, season] and applies the selected
    weighting method. Ensures weights sum to 1.0 per group.

    Parameters
    ----------
    skill_df : pd.DataFrame
        Skill metrics dataframe output by SkillComputer.compute().
    method : str
        The weighting method to use: 'inverse_rmse' or 'rank'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with an added 'weight' column.
        
    Raises
    ------
    ValueError
        If the weights for any group fail to sum to 1.0 (within 1e-6 tolerance).
    """
    df = skill_df.copy()
    df["weight"] = 0.0
    
    group_cols = ["variable", "region", "lead_hours", "season"]
    
    def calc_group_weights(rmse_series: pd.Series) -> np.ndarray:
        weights = np.zeros(len(rmse_series))
        
        if method == "inverse_rmse":
            rmses = rmse_series.values
            
            # Handle edge case where RMSE is exactly 0
            if np.any(rmses == 0):
                mask = (rmses == 0)
                weights[mask] = 1.0 / np.sum(mask)
            else:
                inv_rmse = 1.0 / rmses
                weights = inv_rmse / np.sum(inv_rmse)
                
        elif method == "rank":
            # Rank models ascending by RMSE (1 = best/lowest RMSE)
            ranks = rmse_series.rank(method="min").values
            inv_rank = 1.0 / ranks
            weights = inv_rank / np.sum(inv_rank)
            
        else:
            raise ValueError(f"Unknown weight method: {method}")
            
        # Assertion: weights must sum to 1.0 per group
        total_weight = np.sum(weights)
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights do not sum to 1.0. Sum: {total_weight}")
            
        return weights

    # Apply calculation group by group using transform on the rmse column
    df["weight"] = df.groupby(group_cols)["rmse"].transform(calc_group_weights)
    
    return df
