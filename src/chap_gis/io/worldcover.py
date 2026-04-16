"""ESA WorldCover 10 m land-cover loader."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
import openeo

from .cache import cache_dir

# borrowing some things from dhis2eo for later integration
from dhis2eo.utils.types import BBox, DateLike
from dhis2eo.data.utils import force_logging

logger = logging.getLogger(__name__)
force_logging(logger)

def fetch_from_openeo(year, bbox, save_path):
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

def download(
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
            fetch_from_openeo(year, bbox, save_path)

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
