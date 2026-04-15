"""Elevation (Copernicus 30m DEM) loader."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from urllib.request import urlretrieve

import rioxarray
import xarray as xr
import pystac_client
import fsspec
from dotenv import load_dotenv

from .cache import cache_dir

# add env vars
load_dotenv()
S3_ACCESS_KEY = os.getenv('COPERNICUS_S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('COPERNICUS_S3_SECRET_KEY')

# borrowing some things from dhis2eo for later integration
from dhis2eo.utils.types import BBox
from dhis2eo.data.utils import force_logging

logger = logging.getLogger(__name__)
force_logging(logger)

# TODO: this should be added to dhis2eo later
def download(
    bbox: BBox, 
    dirname: str,
    prefix: str,
    overwrite: bool = False,
):
    """
    Retrieves Copernicus 30m DEM elevation tile data for a given bbox.
    Saves tile files to disk, as specified by dirname and prefix.
    """
    os.makedirs(dirname, exist_ok=True)

    # connect and authenticate with s3 storage
    s3_url = "https://eodata.dataspace.copernicus.eu"
    logger.info(f'Connecting to s3 {s3_url}')
    fs = fsspec.filesystem(
        "s3",
        client_kwargs={"endpoint_url": s3_url},
        key=S3_ACCESS_KEY,
        secret=S3_SECRET_KEY,
    )

    # connect to copernicus stac catalog
    stac_url = "https://stac.dataspace.copernicus.eu/v1"
    logger.info(f'Connecting to STAC {stac_url}')
    catalog = pystac_client.Client.open(stac_url)

    # find all tiles for given bbox
    collection_id = "cop-dem-glo-30-dged-cog"
    search = catalog.search(
        collections=[collection_id],
        bbox=bbox,
    )

    # process each tile
    files = []
    for item in search.items():
        logger.info(f'Tile {item}')

        # Determine the save path
        fs_path = item.assets['data'].href.replace("s3://", "")
        filename = Path(fs_path).stem
        save_file = f'{prefix}_{filename}.tif'
        save_path = (Path(dirname) / save_file).resolve()
        files.append(save_path)

        # Download or use existing file
        if overwrite is False and save_path.exists():
            # File already exist, load from file instead
            logger.info(f'File already downloaded: {save_path}')

        else:
            # Download the data
            logger.info(f'Downloading file {filename} to {save_path}')
            fs.get(fs_path, save_path)
    
    # Return downloaded files
    return files

def load(
        bbox: list,
    ) -> xr.DataArray:
    """Load a digital elevation model (DEM) as a dask-backed DataArray on its native grid.

    The file's CRS is preserved on the returned DataArray. Nodata pixels are
    converted to NaN. 
    """
    # get files from cache or download
    files = download(
        bbox=bbox, 
        dirname=cache_dir(), 
        prefix='copernicus_elevation',
    )

    # open as xarray
    ds = xr.open_mfdataset(files)
    da = ds['band_data'].squeeze()

    # add metadata
    da.name = "elevation"
    da.attrs.update(
        long_name="Terrain elevation above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    )

    return da
