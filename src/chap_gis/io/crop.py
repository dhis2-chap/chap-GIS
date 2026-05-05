"""Crop extent (NASA LGRIP30) loader.

This script requires registering with NASA EarthData, generating a Bearer Token,
and adding it to environment variables as NASA_EARTHDATA_TOKEN.

https://urs.earthdata.nasa.gov/documentation/for_users/user_token
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from geopandas import GeoDataFrame
import xarray as xr
import rioxarray
from rioxarray.merge import merge_arrays
import pystac_client
import fsspec
from dotenv import load_dotenv

from .cache import cache_dir
from ._naming import dataset_prefix

# borrowing some things from dhis2eo for later integration
from dhis2eo.utils.types import BBox, DateLike
from dhis2eo.data.utils import force_logging

load_dotenv()
HEADER_TOKEN = os.getenv('NASA_EARTHDATA_TOKEN')

logger = logging.getLogger(__name__)
force_logging(logger)


dataset_id = "lgrip_crop_extent"


# TODO: this should be added to dhis2eo later
def save_to_file(fs, fs_path, save_path):
    logger.info(f'Downloading file {fs_path} to {save_path}')
    fs.get(fs_path, save_path)


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
    """Retrieve NASA LGRIP30 crop tiles intersecting ``bbox``.

    ``start``/``end`` are accepted for protocol symmetry but ignored — LGRIP30
    is a static product.
    """
    if bbox is None:
        raise TypeError("crop.download requires bbox")
    dirname = Path(dirname or cache_dir())
    dirname.mkdir(parents=True, exist_ok=True)
    prefix = prefix or dataset_prefix(country_code, dataset_id)

    fs = fsspec.filesystem(
        "https",
        headers={"Authorization": f"Bearer {HEADER_TOKEN}"},
    )

    stac_url = "https://cmr.earthdata.nasa.gov/stac/LPDAAC_ECS"
    logger.info(f'Connecting to STAC {stac_url}')
    catalog = pystac_client.Client.open(stac_url)

    collection_id = "LGRIP30_001"
    search = catalog.search(
        collections=[collection_id],
        bbox=bbox,
    )

    files = []
    for item in search.items():
        logger.info(f'Tile {item}')

        # Tif file is under a weird string key, so best to just find the first .tif string
        fs_path = [asset.href for asset in item.assets.values() if asset.href.endswith('.tif')][0]

        filename = Path(fs_path).name
        save_file = f'{prefix}_{filename}'
        save_path = (dirname / save_file).resolve()
        files.append(save_path)

        if overwrite is False and save_path.exists():
            logger.info(f'File already downloaded: {save_path}')
        else:
            save_to_file(fs, fs_path, save_path)

    return files


def load(
    aoi: GeoDataFrame,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    country_code: str | None = None,
) -> xr.DataArray:
    """Load NASA LGRIP30 crop extent as a dask-backed DataArray on its native grid.

    ``start``/``end`` are accepted for protocol symmetry but ignored — LGRIP30
    is a static product.
    """
    bbox = list(map(float, aoi.total_bounds))

    files = download(bbox=bbox, country_code=country_code)

    lazy_arrays = [rioxarray.open_rasterio(fil, chunks={"x": 2048, "y": 2048}) for fil in files]
    da = merge_arrays(lazy_arrays)
    da = da.squeeze('band')

    # TODO: select only where crop is?
    # 0=nd, 1=nocrop, 2=crop

    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    da.name = "crop"
    da.attrs.update(
        long_name="Crop extent",
        standard_name="area_fraction",
        units="1",
        source="NASA LGRIP30 v001",
    )

    return da
