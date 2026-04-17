"""Crop extent (NASA 30m) loader."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from geopandas import GeoDataFrame
import xarray as xr
import pystac_client
import fsspec
from dotenv import load_dotenv

from .cache import cache_dir

# add env vars
load_dotenv()
HEADER_TOKEN = os.getenv('NASA_EARTHDATA_TOKEN')

# borrowing some things from dhis2eo for later integration
from dhis2eo.utils.types import BBox
from dhis2eo.data.utils import force_logging

logger = logging.getLogger(__name__)
force_logging(logger)

# TODO: this should be added to dhis2eo later
def save_to_file(fs, fs_path, save_path):
    logger.info(f'Downloading file {fs_path} to {save_path}')
    fs.get(fs_path, save_path)

def download(
    bbox: BBox,
    dirname: str,
    prefix: str,
    overwrite: bool = False,
) -> list[Path]:
    """
    Retrieves NASA GFSAD/LGRIP 30m crop tile data for a given bbox.
    Saves tile files to disk, as specified by dirname and prefix.
    """
    os.makedirs(dirname, exist_ok=True)

    # connect and add bearer token

    fs = fsspec.filesystem(
        "https",
        headers={
            "Authorization": f"Bearer {HEADER_TOKEN}"
        }
    )

    # connect to copernicus stac catalog
    stac_url = "https://cmr.earthdata.nasa.gov/stac/LPDAAC_ECS"
    logger.info(f'Connecting to STAC {stac_url}')
    catalog = pystac_client.Client.open(stac_url)

    # find all tiles for given bboxs
    collection_id = "LGRIP30_001"
    search = catalog.search(
        collections=[collection_id],
        bbox=bbox,
    )

    # create pooled downloader
    downloader = ThreadPoolExecutor(max_workers=4)

    # process each tile
    files = []
    for item in search.items():
        logger.info(f'Tile {item}')
        
        # Get the tile url
        # Tif file is under a weird string key, so best to just find the first .tif string
        fs_path = [asset.href for asset in item.assets.values() if asset.href.endswith('.tif')][0]

        # Determine the save path
        filename = Path(fs_path).name
        save_file = f'{prefix}_{filename}'
        save_path = (Path(dirname) / save_file).resolve()
        files.append(save_path)

        # Download or use existing file
        if overwrite is False and save_path.exists():
            # File already exist, load from file instead
            logger.info(f'File already downloaded: {save_path}')

        else:
            # Download the data
            #downloader.submit(save_file, fs, fs_path, save_path)
            save_to_file(fs, fs_path, save_path)

            # TODO: these are large tiles, likely need to save to temporary folder
            # then crop to bbox and save to target location
            # ... 

    # Wait for all downloads to finish
    downloader.shutdown(wait=True)
    
    # Return downloaded files
    return files

def load(
        aoi: GeoDataFrame,
    ) -> xr.DataArray:
    """Load NASA 30m crop extent data as a dask-backed DataArray on its native grid.

    The file's CRS is preserved on the returned DataArray. Nodata pixels are
    converted to NaN. 
    """
    # get bbox from aoi
    bbox = list(map(float, aoi.total_bounds))

    # get files from cache or download
    files = download(
        bbox=bbox,
        dirname=cache_dir(),
        prefix='nasa_crops',
    )

    # merge tiles to single xarray
    import rioxarray
    from rioxarray.merge import merge_arrays
    lazy_arrays = [rioxarray.open_rasterio(fil, chunks={"x": 2048, "y": 2048}) for fil in files]
    da = merge_arrays(lazy_arrays)
    logger.info(da)
    da = da.squeeze('band')

    # TODO: select only where crop is? 
    # 0=nd, 1=nocrop, 2=crop
    # ... 

    # add metadata
    da.name = "crop"
    da.attrs.update(
        long_name="Crop extent",
        units="presence of crop",
    )
    logger.info(da)

    return da
