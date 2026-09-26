"""
src/utils/config_loader.py
--------------------------
Layer   : utilities (cross-cutting)
Purpose : Load YAML configuration files into plain Python dicts.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

import yaml


def load_config(path: str) -> dict:
    """Load a YAML configuration file and return its contents as a dict.

    Parameters
    ----------
    path : str
        Relative or absolute path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration as a nested Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist at *path*.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'. "
            "Ensure the file exists and the path is correct relative to the "
            "project root."
        )
