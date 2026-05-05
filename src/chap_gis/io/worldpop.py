"""WorldPop population loader."""

from __future__ import annotations

from pathlib import Path

import xarray as xr
from dhis2eo.data.worldpop import pop_total
from dhis2eo.utils.types import DateLike
from geopandas import GeoDataFrame

from .cache import cache_dir, cache_key


dataset_id = "worldpop_population_yearly"


def download(
    start: DateLike,
    end: DateLike,
    *,
    country_code: str,
    dirname: str | Path | None = None,
    prefix: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download WorldPop yearly population rasters for ``country_code``.

    Delegates to :func:`dhis2eo.data.worldpop.pop_total.yearly.download`.
    """
    dirname = Path(dirname or cache_dir())
    dirname.mkdir(parents=True, exist_ok=True)
    prefix = prefix or cache_key(dataset_id, country_code)

    return pop_total.yearly.download(
        start=str(start),
        end=str(end),
        country_code=country_code,
        dirname=dirname,
        prefix=prefix,
        overwrite=overwrite,
    )


def load(
    aoi: GeoDataFrame | None = None,
    *,
    start: DateLike,
    end: DateLike,
    country_code: str,
) -> xr.DataArray:
    """Load a WorldPop people-per-pixel raster as a lazy DataArray.

    ``aoi`` is accepted for protocol symmetry but ignored — WorldPop is fetched
    by ``country_code`` (ISO3) directly.
    """
    files = download(
        start=start,
        end=end,
        country_code=country_code,
    )

    ds = xr.open_mfdataset(files)
    da = ds['pop_total']
    encoding = da.encoding

    da = da.fillna(0)
    da = da.astype('float32')

    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    da.name = "population"
    da.attrs.update(
        long_name="Population count per pixel",
        standard_name="number_concentration_of_people_in_air",
        units="people",
        source=f"WorldPop {start}..{end}",
    )
    da.encoding = encoding

    return da
