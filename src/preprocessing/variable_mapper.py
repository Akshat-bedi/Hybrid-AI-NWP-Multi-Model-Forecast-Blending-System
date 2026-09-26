"""
src/preprocessing/variable_mapper.py
------------------------------------
Layer   : preprocessing
Purpose : Renames variables to standard names and converts units.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import ast
import logging

import xarray as xr

logger = logging.getLogger(__name__)


class VariableMapper:
    """Maps dataset variables to standard names and converts units."""

    PRECIP_KGMS_TO_MMHR: float = 3600.0   # kg/m2/s -> mm/hr
    TEMP_C_TO_K: float = 273.15           # degC -> K

    def map(self, ds: xr.Dataset, model_name: str, model_config: dict) -> xr.Dataset:
        """Rename variables and apply unit conversions based on configuration.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to map.
        model_name : str
            The name or key of the model.
        model_config : dict
            The configuration dictionary containing all models or the specific model.

        Returns
        -------
        xr.Dataset
            The dataset with mapped variables and standard units.
        """
        # Accommodate both full registry dict or a specific model's dict
        if model_name in model_config:
            cfg = model_config[model_name]
        else:
            cfg = model_config

        var_mapping = cfg.get("var_mapping", {})
        source_units = cfg.get("source_units", {})

        # Rename variables
        rename_dict = {
            src: dst 
            for src, dst in var_mapping.items() 
            if src in ds.data_vars
        }
        if rename_dict:
            ds = ds.rename(rename_dict)
            logger.debug("Renamed variables: %s", rename_dict)

        # Parse current units
        current_units = ds.attrs.get("units", "{}")
        if isinstance(current_units, str):
            try:
                units_dict = ast.literal_eval(current_units)
            except (ValueError, SyntaxError):
                units_dict = {}
        else:
            units_dict = dict(current_units)

        # Apply unit conversions
        for var in ["t2m", "tp", "u10", "v10", "mslp"]:
            if var not in ds.data_vars:
                continue
                
            src_unit = source_units.get(var, "").lower()
            
            # Precipitation conversion
            if var == "tp" and src_unit in ["kg/m2/s", "kg m-2 s-1", "kg/m^2/s", "kgms"]:
                ds[var] = ds[var] * self.PRECIP_KGMS_TO_MMHR
                units_dict[var] = "mm/hr"
                logger.debug("Converted tp from kg/m2/s to mm/hr")
            elif var == "tp" and not src_unit:
                units_dict[var] = "mm/hr"

            # Temperature conversion
            elif var == "t2m" and src_unit in ["c", "celsius", "degc", "deg c", "°c"]:
                ds[var] = ds[var] + self.TEMP_C_TO_K
                units_dict[var] = "K"
                logger.debug("Converted t2m from Celsius to Kelvin")
            elif var == "t2m" and not src_unit:
                units_dict[var] = "K"

            # Update others to standard without conversion if not specified
            elif var in ["u10", "v10"] and not src_unit:
                units_dict[var] = "m/s"
            elif var == "mslp" and not src_unit:
                units_dict[var] = "Pa"

        ds.attrs["units"] = str(units_dict)
        return ds
