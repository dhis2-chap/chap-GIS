"""Elevation (Copernicus 30m DEM) loader.

This script requires registering with Copernicus Data Space Ecosystem (CDSE), 
generating an OAuth Client, and adding it to environment variables. 

See README.md in the root folder for instructions. 
"""

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
import openeo
from dotenv import load_dotenv

from .cache import cache_dir

# borrowing some things from dhis2eo for later integration
from dhis2eo.utils.types import BBox
from dhis2eo.data.utils import force_logging

logger = logging.getLogger(__name__)
force_logging(logger)

# add env vars
load_dotenv()
CDSE_OAUTH_CLIENT_ID = os.getenv('CDSE_OAUTH_CLIENT_ID')
CDSE_OAUTH_CLIENT_SECRET = os.getenv('CDSE_OAUTH_CLIENT_SECRET')

# TODO: below should be added to dhis2eo later

##################
# openeo approach

def fetch_openeo(bbox, save_path):
    # Note: this function uses openeo which requires authentication keys and can take up to 5 mins

    # connect to openeo
    # see module docstring for instructions on getting credentials
    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc_client_credentials(
        client_id=CDSE_OAUTH_CLIENT_ID,
        client_secret=CDSE_OAUTH_CLIENT_SECRET,
        #provider_id='...' # not needed?
    )

    # create bbox dict
    xmin,ymin,xmax,ymax = bbox
    bbox_dict = {
        "west": xmin,
        "south": ymin,
        "east": xmax,
        "north": ymax,
    }

    # load collection
    coll_id = 'COPERNICUS_30'
    logger.info(f"Loading collection: {coll_id} {bbox_dict}")
    wc = conn.load_collection(
        coll_id,
        spatial_extent=bbox_dict,
    )

    # copernicus dem has readings from multiple dates
    # need to take the mean to get a single file
    wc = wc.reduce_dimension(
        dimension="t",
        reducer="mean",
    )

    # schedule saving to file
    #wc.save_result(format="NetCDF")  # silently ignores saving as netcdf for some reason
    wc.save_result(format="GeoTIFF")

    # submit as asynch job
    # and wait for job to finish
    logger.info('Waiting for job data request...')
    job = wc.execute_batch(title="Retrieve elevation for bbox")  # waits for result to finish

    # validate results
    if job.status() == 'finished':
        # download to disk
        logger.info(f'Job finished, downloading results...')
        results = job.get_results()
        results.download_file(save_path)
    
    else:
        raise Exception(f'Failed to retrieve data from openeo service: {job.status()}')


def download(
    bbox: BBox,
    dirname: str,
    prefix: str,
    overwrite: bool = False,
) -> list[Path]:
    """
    Retrieves Copernicus 30m DEM elevation tile data for a given bbox.
    Saves tile files to disk, as specified by dirname and prefix.
    """
    os.makedirs(dirname, exist_ok=True)

    # Determine the save path
    save_file = f'{prefix}.tif'
    save_path = (Path(dirname) / save_file).resolve()
    files = [save_path]

    # Download or use existing file
    if overwrite is False and save_path.exists():
        # File already exist, load from file instead
        logger.info(f'File already downloaded: {save_path}')

    else:
        # Download the data
        fetch_openeo(bbox, save_path)

    return files


###################
# aws s3 approach

# def save_s3_file(fs, fs_path, save_path):
#     logger.info(f'Downloading file {fs_path} to {save_path}')
#     fs.get(fs_path, save_path)

# def download_s3(
#     bbox: BBox,
#     dirname: str,
#     prefix: str,
#     overwrite: bool = False,
# ) -> list[Path]:
#     """
#     Retrieves Copernicus 30m DEM elevation tile data for a given bbox.
#     Saves tile files to disk, as specified by dirname and prefix.
#     """
#     os.makedirs(dirname, exist_ok=True)

#     # connect and authenticate with s3 storage
#     s3_url = "https://eodata.dataspace.copernicus.eu"
#     logger.info(f'Connecting to s3 {s3_url}')
#     fs = fsspec.filesystem(
#         "s3",
#         client_kwargs={"endpoint_url": s3_url},
#         key=S3_ACCESS_KEY,
#         secret=S3_SECRET_KEY,
#     )

#     # connect to copernicus stac catalog
#     stac_url = "https://stac.dataspace.copernicus.eu/v1"
#     logger.info(f'Connecting to STAC {stac_url}')
#     catalog = pystac_client.Client.open(stac_url)

#     # find all tiles for given bbox
#     collection_id = "cop-dem-glo-30-dged-cog"
#     search = catalog.search(
#         collections=[collection_id],
#         bbox=bbox,
#     )

#     # create pooled downloader
#     downloader = ThreadPoolExecutor(max_workers=10)

#     # process each tile
#     files = []
#     for item in search.items():
#         logger.info(f'Tile {item}')

#         # Determine the save path
#         fs_path = item.assets['data'].href.replace("s3://", "")
#         filename = Path(fs_path).stem
#         save_file = f'{prefix}_{filename}.tif'
#         save_path = (Path(dirname) / save_file).resolve()
#         files.append(save_path)

#         # Download or use existing file
#         if overwrite is False and save_path.exists():
#             # File already exist, load from file instead
#             logger.info(f'File already downloaded: {save_path}')

#         else:
#             # Download the data
#             downloader.submit(save_s3_file, fs, fs_path, save_path)
        
#         # Brief pause to avoid overwhelming stac service
#         time.sleep(0.3)

#     # Wait for all downloads to finish
#     downloader.shutdown(wait=True)
    
#     # Return downloaded files
#     return files

def load(
        aoi: GeoDataFrame,
    ) -> xr.DataArray:
    """Load a digital elevation model (DEM) as a dask-backed DataArray on its native grid.

    The file's CRS is preserved on the returned DataArray. Nodata pixels are
    converted to NaN. 
    """
    # get bbox from aoi
    bbox = list(map(float, aoi.total_bounds))

    # get files from cache or download
    files = download(
        bbox=bbox,
        dirname=cache_dir(),
        prefix='copernicus_elevation',
    )

    # open as xarray
    ds = xr.open_dataset(files[0])

    # convert to dataarray
    da = ds['band_data'].squeeze('band')

    # make it spatial
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # add metadata
    da.name = "elevation"
    da.attrs.update(
        long_name="Terrain elevation above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    )

    return da
