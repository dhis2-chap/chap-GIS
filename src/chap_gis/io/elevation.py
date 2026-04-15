"""Elevation (SRTM / generic DEM) loader."""

from __future__ import annotations

from pathlib import Path

import rioxarray
import xarray as xr


def load(path: str | Path, chunks: str | dict = "auto") -> xr.DataArray:
    """Load a DEM GeoTIFF as a dask-backed DataArray on its native grid.

    The file's CRS is preserved on the returned DataArray. Nodata pixels are
    converted to NaN. ``chunks`` is passed through to
    :func:`rioxarray.open_rasterio`; defaults to ``"auto"`` for laziness.
    """
    da = rioxarray.open_rasterio(path, masked=True, chunks=chunks).squeeze(
        "band", drop=True
    )
    da = da.astype("float32")
    da.name = "elevation"
    da.attrs.update(
        long_name="Terrain elevation above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    )
    return da
