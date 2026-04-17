"""Rice field Africa (20m) loader."""

# NOTE: For alternative rice datasets and regions, see:
# https://www.sciencedirect.com/science/article/pii/S003442572600026X#bib508

from __future__ import annotations

import logging

import xarray as xr

from .cache import cache_dir

def load(
        iso3: str,
    ) -> xr.DataArray:
    """Load 20m Africa rice distribution data for 2023 (Jiang et al) as a dask-backed DataArray on its native grid.

    The file's CRS is preserved on the returned DataArray. Nodata pixels are
    converted to NaN. 
    """
    # open from already downloaded data
    path = cache_dir().parent / 'inputs' / f'rice_fields_{iso3.lower()}.tif'
    da = xr.open_dataarray(path)
    da = da.squeeze('band')

    # add metadata
    da.name = "rice"
    da.attrs.update(
        long_name="Rice fields",
        units="presence of rice fields",
    )
    logging.info(da)

    return da
