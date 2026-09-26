"""
src/preprocessing/regridder.py
------------------------------
Layer   : preprocessing
Purpose : Regrid xr.Datasets to the standard India domain grid.

Part of the Hybrid AI-NWP Multi-Model Forecast Blending System
(SIH Problem 26081 | NCMRWF / MoES)
"""

from __future__ import annotations

import logging
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

TARGET_LAT = np.arange(6.0, 38.25, 0.25)
TARGET_LON = np.arange(68.0, 97.25, 0.25)


class Regridder:
    """Regrids datasets to the standard target grid."""

    def __init__(
        self, target_lat: np.ndarray = TARGET_LAT, target_lon: np.ndarray = TARGET_LON
    ) -> None:
        """Initialize the Regridder with target latitude and longitude arrays."""
        self.target_lat = target_lat
        self.target_lon = target_lon

    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        """Regrid all variables in the dataset to the target grid.

        Uses linear interpolation via scipy.interpolate.RegularGridInterpolator.
        Missing data outside the source domain is filled with NaN.
        
        Parameters
        ----------
        ds : xr.Dataset
            The source dataset to regrid.

        Returns
        -------
        xr.Dataset
            A new dataset on the target grid with preserved attributes.

        Raises
        ------
        ValueError
            If source grid spacing is > 1.0 degree.
        """
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError("Dataset must contain 'lat' and 'lon' coordinates.")

        src_lat = ds.lat.values
        src_lon = ds.lon.values

        if len(src_lat) > 1:
            lat_spacing = np.abs(src_lat[1] - src_lat[0])
            if lat_spacing > 1.0:
                raise ValueError(f"Source lat spacing ({lat_spacing:.2f}) > 1.0 deg.")
        
        if len(src_lon) > 1:
            lon_spacing = np.abs(src_lon[1] - src_lon[0])
            if lon_spacing > 1.0:
                raise ValueError(f"Source lon spacing ({lon_spacing:.2f}) > 1.0 deg.")

        regridded_vars = {}
        for var_name, da in ds.data_vars.items():
            regridded_vars[str(var_name)] = self._regrid_variable(da)
        
        out_ds = xr.Dataset(regridded_vars)
        out_ds = out_ds.assign_attrs(ds.attrs)
        
        # Copy non-lat/lon coordinates
        for coord_name in ds.coords:
            if coord_name not in ["lat", "lon"]:
                out_ds = out_ds.assign_coords({coord_name: ds.coords[coord_name]})
                
        return out_ds

    def _regrid_variable(self, da: xr.DataArray) -> xr.DataArray:
        """Regrid a single xr.DataArray.
        
        Loops over the lead_hours dimension and regrids each 2D slice.
        Preserves all DataArray attributes.
        """
        src_lat = da.lat.values
        src_lon = da.lon.values

        # RegularGridInterpolator requires strictly ascending coordinates
        flip_lat = src_lat[0] > src_lat[-1]
        flip_lon = src_lon[0] > src_lon[-1]
        
        slat = src_lat[::-1] if flip_lat else src_lat
        slon = src_lon[::-1] if flip_lon else src_lon

        lon_grid, lat_grid = np.meshgrid(self.target_lon, self.target_lat)
        target_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

        out_shape = list(da.shape)
        lat_idx = da.dims.index("lat")
        lon_idx = da.dims.index("lon")
        out_shape[lat_idx] = len(self.target_lat)
        out_shape[lon_idx] = len(self.target_lon)
        
        out_data = np.full(out_shape, np.nan, dtype=np.float32)

        if "lead_hours" in da.dims:
            for i, lh in enumerate(da.coords["lead_hours"].values):
                slice_2d = da.isel(lead_hours=i).values
                
                # Ensure data order matches the ascending coordinate order
                if flip_lat:
                    slice_2d = np.flip(slice_2d, axis=0)
                if flip_lon:
                    slice_2d = np.flip(slice_2d, axis=1)
                
                interp = RegularGridInterpolator(
                    (slat, slon), slice_2d, 
                    method='linear', bounds_error=False, fill_value=np.nan
                )
                regridded = interp(target_points).reshape(len(self.target_lat), len(self.target_lon))
                
                # Assign to the corresponding slice in out_data
                out_data[i, :, :] = regridded
        else:
            slice_2d = da.values
            if flip_lat:
                slice_2d = np.flip(slice_2d, axis=0)
            if flip_lon:
                slice_2d = np.flip(slice_2d, axis=1)
                
            interp = RegularGridInterpolator(
                (slat, slon), slice_2d, 
                method='linear', bounds_error=False, fill_value=np.nan
            )
            regridded = interp(target_points).reshape(len(self.target_lat), len(self.target_lon))
            out_data = regridded

        out_da = xr.DataArray(
            out_data,
            dims=da.dims,
            coords={"lat": self.target_lat, "lon": self.target_lon},
            attrs=da.attrs
        )
        return out_da
