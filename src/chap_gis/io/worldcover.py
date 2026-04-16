"""ESA WorldCover 10 m land-cover loader."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
import openeo
import fsspec

from .cache import cache_dir

# borrowing some things from dhis2eo for later integration
from dhis2eo.utils.types import BBox, DateLike
from dhis2eo.data.utils import force_logging

logger = logging.getLogger(__name__)
force_logging(logger)

##################
# openeo approach

def fetch_year_openeo(year, bbox, save_path):
    # Note: this function uses openeo which requires authentication keys and can take up to 5 mins
    # If needed, we can also explore direct against their S3 buckets which likely will be faster
    # ...and result in multiple downloaded tiles

    # connect to openeo
    # Note: right now requires manual login only the first time
    # TODO: switch to passing client key and secret key from env vars
    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()  # triggers manual login

    # create bbox dict
    xmin,ymin,xmax,ymax = bbox
    bbox_dict = {
        "west": xmin,
        "south": ymin,
        "east": xmax,
        "north": ymax,
    }

    # load collection
    year_to_suffix = {
        2020: '2020_V1',
        2021: '2021_V2',
    }
    suffix = year_to_suffix[year]
    logger.info(f"Loading collection: ESA_WORLDCOVER_10M_{suffix} {bbox_dict}")
    wc = conn.load_collection(
        f"ESA_WORLDCOVER_10M_{suffix}",
        spatial_extent=bbox_dict,
    )

    # schedule saving to netcdf
    wc.save_result(format="NetCDF")

    # submit as asynch job
    # and wait for job to finish
    logger.info('Waiting for job data request...')
    job = wc.execute_batch(title="Retrieve worldcover for bbox")  # waits for result to finish

    # validate results
    if job.status() == 'finished':
        # download to disk
        logger.info(f'Job finished, downloading results...')
        results = job.get_results()
        results.download_file(save_path)
    
    else:
        raise Exception(f'Failed to retrieve data from openeo service: {job.status()}')

def fetch_years_openeo(
    start: DateLike,
    end: DateLike,
    bbox: BBox,
    dirname: str,
    prefix: str,
    overwrite: bool = False,
) -> list[Path]:
    # For every year
    start_year = int(start)
    end_year = int(end)
    files = []
    for year in range(start_year, end_year + 1):
        logger.info(f'Year {year}')

        # Determine the save path
        save_file = f'{prefix}_{year}.tif'
        save_path = (Path(dirname) / save_file).resolve()
        files.append(save_path)

        # Download or use existing file
        if overwrite is False and save_path.exists():
            # File already exist, load from file instead
            logger.info(f'File already downloaded: {save_path}')

        else:
            # Download the data
            logger.info(f'Fetching data for {year}')
            fetch_year_openeo(year, bbox, save_path)

    return files

#################
# aws s3 approach

def save_s3_file(fs, fs_path, save_path):
    logger.info(f'Downloading file {fs_path} to {save_path}')
    fs.get(fs_path, save_path)

def fetch_year_s3(
    year: int,
    bbox: BBox,
    dirname: str,
    prefix: str,
    overwrite: bool = False,
) -> list[Path]:
    """
    Retrieves WorldCover tile data for a given bbox.
    Saves tile files to disk, as specified by dirname and prefix.
    """
    os.makedirs(dirname, exist_ok=True)

    # create geometry from bbox
    from shapely.geometry import box
    geom = box(*bbox)

    # connect and authenticate with s3 storage
    s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    logger.info(f'Connecting to s3 {s3_url_prefix}')
    fs = fsspec.filesystem("https")

    # load worldcover grid geojson
    tile_grid_url = f'{s3_url_prefix}/esa_worldcover_grid.geojson'
    tile_grid = gpd.read_file(tile_grid_url)

    # get grid tiles intersecting AOI
    tiles = tile_grid[tile_grid.intersects(geom)]

    # select version tag, based on the year
    version = {
        2020: 'v100',
        2021: 'v200'
    }[year]

    # create pooled downloader
    downloader = ThreadPoolExecutor(max_workers=10)

    # process each tile
    files = []
    for tile in tiles.ll_tile:
        logger.info(f'Tile {tile}')

        # Determine the save path
        fs_path = f"{s3_url_prefix}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
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
            #downloader.submit(save_s3_file, fs, fs_path, save_path)
            save_s3_file(fs, fs_path, save_path)

            # TODO: these are large tiles, likely need to crop to bbox after download too
            # ... 
        
        # Brief pause to avoid overwhelming service
        #time.sleep(0.3)

    # Wait for all downloads to finish
    downloader.shutdown(wait=True)
    
    # Return downloaded files
    return files

############
# main

def download(
    start: DateLike,
    end: DateLike,
    bbox: BBox,
    dirname: str,
    prefix: str,
    overwrite: bool = False,
) -> list[Path]:
    
    # download from openeo
    files = fetch_years_openeo(
        start,
        end,
        bbox,
        dirname,
        prefix,
        overwrite,
    )

    # download from s3
    # start = int(start)
    # end = int(end)
    # files = []
    # for year in range(start, end + 1):
    #     files += fetch_year_s3(year, bbox, dirname, prefix, overwrite)

    return files

def load(
    aoi: gpd.GeoDataFrame,
    year: int = 2021,
) -> xr.DataArray:
    """
    Retrieve dataset for WorldCover cropped to aoi via openeo
    """
    # get bbox
    if str(aoi.crs) != "EPSG:4326":
        aoi = aoi.to_crs("EPSG:4326")
    bounds = list(map(float, aoi.total_bounds))

    # download and open
    files = download(
        start=year,
        end=year,
        bbox=bounds,
        dirname=cache_dir(),
        prefix='worldcover',
    )
    ds = xr.open_mfdataset(files)
    da = ds['band_data'].squeeze('band')

    # add metadata
    da.name = "landcover"
    da.attrs.update(
        long_name="ESA WorldCover land cover class",
        standard_name="land_cover_lccs_class",
        units="1",
        source=f"ESA WorldCover",
    )
    da.rio.write_crs("EPSG:4326")

    # return
    return da
