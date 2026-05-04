"""CHELSA monthly temperature loader."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import fsspec
from geopandas import GeoDataFrame

from .cache import cache_dir

# borrowing some things from dhis2eo for later integration
from dhis2eo.utils.types import BBox, DateLike
from dhis2eo.utils.time import iter_months
from dhis2eo.data.utils import force_logging


logger = logging.getLogger(__name__)
force_logging(logger)


# def fetch_day(variable, year, month, day):
#     version = 'V.2.1'
#     url = f'https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/{variable}/{year}/CHELSA_{variable}_{day}_{month}_{year}_{version}.tif'


def fetch_month(variable, bbox, year, month, save_path):
    # create url path
    version = 'V.2.1'
    url = f'https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/monthly/{variable}/{year}/CHELSA_{variable}_{str(month).zfill(2)}_{year}_{version}.tif'

    # Connect to global dataset lazily
    da = rioxarray.open_rasterio(
        url,
        chunks=None, # disable dask, not needed and actually slows things down
    )
    
    # Read only the bbox window
    xmin, ymin, xmax, ymax = bbox
    da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
    
    # Ensure nodata value is masked and added to metadata
    #nodata = -9999.0 # this should be the chirps3 nodata value
    #da = da.where(da != nodata) # this adds nans where nodata for plotting
    #da.rio.write_nodata(nodata, encoded=True, inplace=True) # should write to metadata for future saving

    # Convert to dataset
    ds = da.to_dataset(name=variable)

    # Remove unnecessary band dim
    ds = ds.squeeze("band", drop=True)

    # Add month constant
    ds = ds.expand_dims(time=[np.datetime64(f'{year}-{str(month).zfill(2)}-01')])

    # Save to netcdf
    ds.to_netcdf(save_path)


def download(
    start: DateLike,
    end: DateLike,
    bbox: BBox,
    dirname: str,
    prefix: str,
    variable: str,
    overwrite: bool = False,
) -> list[Path]:
    """
    Retrieves CHELSA 1km climate data for a given bbox.
    Saves files to disk, as specified by dirname and prefix.
    """
    os.makedirs(dirname, exist_ok=True)

    # Create multithread downloader
    downloader = ThreadPoolExecutor(max_workers=4)

    # Loop months
    start_year, start_month = map(int, start.split('-'))
    end_year, end_month = map(int, end.split('-'))
    files = []
    for year, month in iter_months(start_year, start_month, end_year, end_month):
        logger.info(f'Month {year}-{month}')

        # Determine the save path
        save_file = f'{prefix}_{year}-{str(month).zfill(2)}.tif'
        save_path = (Path(dirname) / save_file).resolve()
        files.append(save_path)

        # Download or use existing file
        if overwrite is False and save_path.exists():
            # File already exist, load from file instead
            logger.info(f'File already downloaded: {save_path}')

        else:
            # Download the data
            downloader.submit(fetch_month, variable, bbox, year, month, save_path)
            #fetch_month(variable, bbox, year, month, save_path)

    # Wait for remaining downloads
    downloader.shutdown(wait=True)

    return files


def load_monthly_tas(
    aoi: GeoDataFrame,
    year: int,
    country: str | None = None,
) -> xr.DataArray:
    """Load monthly CHELSA near-surface air temperature rasters for a given year.

    Returns a lazy DataArray with dims ``(time, y, x)`` in degrees Celsius.
    """
    # get bbox from aoi
    bbox = list(map(float, aoi.total_bounds))

    # get files from cache or download
    variable = 'tas'  # temperature
    prefix = f'{country}_chelsa_temperature' if country and country.strip() else 'chelsa_temperature'
    files = download(
        start=f'{year}-01',
        end=f'{year}-12',
        bbox=bbox,
        dirname=cache_dir(),
        prefix=prefix,
        variable=variable,
    )

    # open as multifile
    ds = xr.open_mfdataset(files)

    # convert kelvin to celsius
    ds[variable] -= 273.15

    # only return data array
    da = ds[variable]

    # make it spatial
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # add metadata
    da.name = variable
    da.attrs.update(
        long_name="Near-surface air temperature (monthly mean)",
        standard_name="air_temperature",
        units="degC",
        source=f"CHELSA v2.1 monthly tas {year}",
    )
    return da
