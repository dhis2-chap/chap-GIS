"""CHELSA monthly temperature loader."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from geopandas import GeoDataFrame

from .cache import cache_dir, cache_key, cached_download

# borrowing some things from dhis2eo for later integration
from dhis2eo.utils.types import BBox, DateLike
from dhis2eo.utils.time import iter_months
from dhis2eo.data.utils import force_logging


logger = logging.getLogger(__name__)
force_logging(logger)


dataset_id = "chelsa_temperature_monthly"
_VARIABLE = "tas"


def fetch_month(variable, bbox, year, month, save_path):
    version = 'V.2.1'
    url = f'https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/monthly/{variable}/{year}/CHELSA_{variable}_{str(month).zfill(2)}_{year}_{version}.tif'

    da = rioxarray.open_rasterio(
        url,
        chunks=None,  # disable dask, not needed and actually slows things down
    )

    xmin, ymin, xmax, ymax = bbox
    da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)

    ds = da.to_dataset(name=variable)
    ds = ds.squeeze("band", drop=True)
    ds = ds.expand_dims(time=[np.datetime64(f'{year}-{str(month).zfill(2)}-01')])
    ds.to_netcdf(save_path)


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
    """Retrieve CHELSA 1km monthly tas rasters for ``bbox`` between ``start`` and ``end``.

    ``start``/``end`` are ``YYYY-MM`` strings. Files are written under
    ``dirname`` with names ``{prefix}_{year}-{month}.tif``. If ``prefix`` is
    omitted it is derived from ``country_code`` and the module ``dataset_id``.
    """
    dirname = Path(dirname or cache_dir())
    prefix = prefix or cache_key(dataset_id, country_code)

    start_year, start_month = map(int, str(start).split('-'))
    end_year, end_month = map(int, str(end).split('-'))
    items = list(iter_months(start_year, start_month, end_year, end_month))

    return cached_download(
        items,
        lambda ym, path: fetch_month(_VARIABLE, bbox, ym[0], ym[1], path),
        dirname=dirname,
        name_fn=lambda ym: f"{prefix}_{ym[0]}-{ym[1]:02d}.tif",
        overwrite=overwrite,
        parallel=True,
        max_workers=4,
        log=logger,
    )


def load(
    aoi: GeoDataFrame,
    *,
    start: DateLike,
    end: DateLike,
    country_code: str | None = None,
) -> xr.DataArray:
    """Load monthly CHELSA near-surface air temperature rasters.

    ``start``/``end`` are ``YYYY-MM`` strings (or any value accepted by
    :class:`dhis2eo.utils.types.DateLike`). Returns a lazy DataArray with dims
    ``(time, y, x)`` in degrees Celsius.
    """
    bbox = list(map(float, aoi.total_bounds))

    files = download(
        start=start,
        end=end,
        bbox=bbox,
        country_code=country_code,
    )

    ds = xr.open_mfdataset(sorted(files), combine="nested",concat_dim="time",join="override",coords="minimal",compat="override")
                        
    ds[_VARIABLE] -= 273.15
    da = ds[_VARIABLE]

    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    da.name = _VARIABLE
    da.attrs.update(
        long_name="Near-surface air temperature (monthly mean)",
        standard_name="air_temperature",
        units="degC",
        source=f"CHELSA v2.1 monthly tas {start}..{end}",
    )
    return da
