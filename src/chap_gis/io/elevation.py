"""Elevation (Copernicus 30m DEM) loader.

This script requires registering with Copernicus Data Space Ecosystem (CDSE),
generating an OAuth Client, and adding it to environment variables.

See README.md in the root folder for instructions.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from geopandas import GeoDataFrame
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


dataset_id = "copernicus_dem_30m"


def fetch_openeo(bbox, save_path):
    # Note: this function uses openeo which requires authentication keys and can take up to 5 mins.

    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc_client_credentials(
        client_id=CDSE_OAUTH_CLIENT_ID,
        client_secret=CDSE_OAUTH_CLIENT_SECRET,
    )

    xmin, ymin, xmax, ymax = bbox
    bbox_dict = {"west": xmin, "south": ymin, "east": xmax, "north": ymax}

    coll_id = 'COPERNICUS_30'
    logger.info(f"Loading collection: {coll_id} {bbox_dict}")
    wc = conn.load_collection(
        coll_id,
        spatial_extent=bbox_dict,
    )

    # copernicus dem has readings from multiple dates - take the mean to get a single file
    wc = wc.reduce_dimension(dimension="t", reducer="mean")

    wc.save_result(format="GTiff")

    logger.info('Waiting for job data request...')
    job = wc.execute_batch(title="Retrieve elevation for bbox")

    if job.status() == 'finished':
        logger.info('Job finished, downloading results...')
        results = job.get_results()
        results.download_file(save_path)
    else:
        raise Exception(f'Failed to retrieve data from openeo service: {job.status()}')


def download(
    start: DateLike | None = None,
    end: DateLike | None = None,
    bbox: BBox | None = None,
    *,
    dirname: str | Path | None = None,
    prefix: str | None = None,
    country_code: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Retrieve Copernicus 30m DEM for ``bbox``.

    ``start``/``end`` are accepted for protocol symmetry but ignored — the DEM
    is a static product. Returns a single-file list.
    """
    if bbox is None:
        raise TypeError("elevation.download requires bbox")
    dirname = Path(dirname or cache_dir())
    dirname.mkdir(parents=True, exist_ok=True)
    prefix = prefix or dataset_prefix(country_code, dataset_id)

    save_file = f'{prefix}.tif'
    save_path = (dirname / save_file).resolve()
    files = [save_path]

    if overwrite is False and save_path.exists():
        logger.info(f'File already downloaded: {save_path}')
    else:
        fetch_openeo(bbox, save_path)

    return files


def load(
    aoi: GeoDataFrame,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    country_code: str | None = None,
) -> xr.DataArray:
    """Load a Copernicus 30m DEM as a dask-backed DataArray on its native grid.

    ``start``/``end`` are accepted for protocol symmetry but ignored — the DEM
    is a static product.
    """
    bbox = list(map(float, aoi.total_bounds))

    files = download(bbox=bbox, country_code=country_code)

    ds = xr.open_dataset(files[0])
    da = ds['band_data'].squeeze('band')

    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    da.name = "elevation"
    da.attrs.update(
        long_name="Terrain elevation above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
        source="Copernicus GLO-30 DEM",
    )

    return da
