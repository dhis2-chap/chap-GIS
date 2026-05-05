"""Rice field Africa (20m) loader."""

# NOTE: For alternative rice datasets and regions, see:
# https://www.sciencedirect.com/science/article/pii/S003442572600026X#bib508

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr
from geopandas import GeoDataFrame

from .cache import cache_dir
from dhis2eo.utils.types import BBox, DateLike


dataset_id = "jiang_rice_fields"


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
    """Rice rasters are pre-staged inputs; there is nothing to download.

    Provided so the module satisfies the :class:`DataSource` protocol. Always
    raises :class:`NotImplementedError`.
    """
    raise NotImplementedError(
        f"{dataset_id} rasters are pre-staged inputs; place them under "
        "data/inputs/rice_fields_{iso3}.tif before calling load()."
    )


def load(
    aoi: GeoDataFrame | None = None,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    country_code: str,
) -> xr.DataArray:
    """Load 20m Africa rice distribution data for 2023 (Jiang et al) as a dask-backed DataArray.

    Reads from the pre-staged file ``data/inputs/rice_fields_{country_code}.tif``.
    ``aoi``, ``start``, ``end`` are accepted for protocol symmetry but ignored.
    """
    iso3 = country_code.lower()
    path = cache_dir().parent / 'inputs' / f'rice_fields_{iso3}.tif'
    da = xr.open_dataarray(path)
    da = da.squeeze('band')

    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    da.name = "rice"
    da.attrs.update(
        long_name="Rice fields",
        standard_name="area_fraction",
        units="1",
        source="Jiang et al. 2023 Africa rice fields 20 m",
    )
    logging.info(da)

    return da
