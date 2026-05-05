"""ESA WorldCover 10 m land-cover loader.

This script requires registering with Copernicus Data Space Ecosystem (CDSE),
generating an OAuth Client, and adding it to environment variables.

See README.md in the root folder for instructions.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import geopandas as gpd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
import openeo
from dotenv import load_dotenv

from .cache import cache_dir
from ._naming import dataset_prefix

# borrowing some things from dhis2eo for later integration
from dhis2eo.utils.types import BBox, DateLike
from dhis2eo.data.utils import force_logging

logger = logging.getLogger(__name__)
force_logging(logger)

load_dotenv()
CDSE_OAUTH_CLIENT_ID = os.getenv('CDSE_OAUTH_CLIENT_ID')
CDSE_OAUTH_CLIENT_SECRET = os.getenv('CDSE_OAUTH_CLIENT_SECRET')


dataset_id = "worldcover_landcover_yearly"


def fetch_year_openeo(year, bbox, save_path):
    # Note: this function uses openeo which requires authentication keys and can take up to 5 mins.

    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc_client_credentials(
        client_id=CDSE_OAUTH_CLIENT_ID,
        client_secret=CDSE_OAUTH_CLIENT_SECRET,
    )

    xmin, ymin, xmax, ymax = bbox
    bbox_dict = {"west": xmin, "south": ymin, "east": xmax, "north": ymax}

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

    wc.save_result(format="NetCDF")

    logger.info('Waiting for job data request...')
    job = wc.execute_batch(title="Retrieve worldcover for bbox")

    if job.status() == 'finished':
        logger.info('Job finished, downloading results...')
        results = job.get_results()
        results.download_file(save_path)
    else:
        raise Exception(f'Failed to retrieve data from openeo service: {job.status()}')


def download(
    start: DateLike,
    end: DateLike,
    bbox: BBox,
    *,
    dirname: str | Path | None = None,
    prefix: str | None = None,
    country_code: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Retrieve ESA WorldCover yearly land-cover rasters for ``bbox``.

    ``start``/``end`` are years (``YYYY`` strings or ints). Files are written
    under ``dirname`` with names ``{prefix}_{year}.tif``. If ``prefix`` is
    omitted it is derived from ``country_code`` and the module ``dataset_id``.
    """
    dirname = Path(dirname or cache_dir())
    dirname.mkdir(parents=True, exist_ok=True)
    prefix = prefix or dataset_prefix(country_code, dataset_id)

    start_year = int(start)
    end_year = int(end)
    files = []
    for year in range(start_year, end_year + 1):
        logger.info(f'Year {year}')

        save_file = f'{prefix}_{year}.tif'
        save_path = (dirname / save_file).resolve()
        files.append(save_path)

        if overwrite is False and save_path.exists():
            logger.info(f'File already downloaded: {save_path}')
        else:
            logger.info(f'Fetching data for {year}')
            fetch_year_openeo(year, bbox, save_path)

    return files


def load(
    aoi: gpd.GeoDataFrame,
    *,
    start: DateLike = 2021,
    end: DateLike = 2021,
    country_code: str | None = None,
) -> xr.DataArray:
    """Load WorldCover land-cover rasters cropped to ``aoi``."""
    if str(aoi.crs) != "EPSG:4326":
        aoi = aoi.to_crs("EPSG:4326")
    bounds = list(map(float, aoi.total_bounds))

    files = download(
        start=start,
        end=end,
        bbox=bounds,
        country_code=country_code,
    )
    ds = xr.open_mfdataset(files)
    da = ds['band_data'].squeeze('band')

    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    da.name = "landcover"
    da.attrs.update(
        long_name="ESA WorldCover land cover class",
        standard_name="land_cover_lccs_class",
        units="1",
        source="ESA WorldCover",
    )

    return da
