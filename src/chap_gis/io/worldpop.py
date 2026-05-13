"""WorldPop population loader."""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr
from dhis2eo.data.worldpop import pop_total
from dhis2eo.utils.types import DateLike
from geopandas import GeoDataFrame

from ..grid import reproject_population_to
from .cache import cache_dir, cache_key

logger = logging.getLogger(__name__)


dataset_id = "worldpop_population_yearly"

# Newer WorldPop release (Global_2015_2030 / R2025A, constrained) only covers
# 2015+. For pre-2015 years we fall back to Global_2000_2020 (UN-adjusted,
# unconstrained). The two datasets use different methodologies, so a time
# series that spans 2014→2015 may show a small step at the boundary.
_GLOBAL1_LAST_YEAR = 2014


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

    Picks the WorldPop release per year: ``Global_2000_2020`` for years
    ≤ 2014, ``Global_2015_2030`` (R2025A) for years ≥ 2015.
    """
    dirname = Path(dirname or cache_dir())
    dirname.mkdir(parents=True, exist_ok=True)
    prefix = prefix or cache_key(dataset_id, country_code)

    start_year, end_year = int(start), int(end)
    files: list[Path] = []

    if start_year <= _GLOBAL1_LAST_YEAR:
        files.extend(pop_total.yearly.download(
            start=str(start_year),
            end=str(min(end_year, _GLOBAL1_LAST_YEAR)),
            country_code=country_code,
            dirname=dirname,
            prefix=prefix,
            version="global1",
            overwrite=overwrite,
        ))
    if end_year > _GLOBAL1_LAST_YEAR:
        files.extend(pop_total.yearly.download(
            start=str(max(start_year, _GLOBAL1_LAST_YEAR + 1)),
            end=str(end_year),
            country_code=country_code,
            dirname=dirname,
            prefix=prefix,
            version="global2",
            overwrite=overwrite,
        ))

    return files


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

    When the requested range spans the 2014/2015 boundary, the older
    ``Global_2000_2020`` rasters (≤2014) are reprojected onto the newer
    ``Global_2015_2030`` grid (≥2015) using sum-preserving resampling so the
    files can be concatenated along time.
    """
    files = download(
        start=start,
        end=end,
        country_code=country_code,
    )

    start_year, end_year = int(start), int(end)
    years = list(range(start_year, end_year + 1))
    assert len(years) == len(files), (
        f"WorldPop year/file count mismatch: {len(years)} years vs {len(files)} files"
    )

    has_global1 = start_year <= _GLOBAL1_LAST_YEAR
    has_global2 = end_year > _GLOBAL1_LAST_YEAR
    if has_global1 and has_global2:
        logger.warning(
            "WorldPop range %d..%d spans two releases with different methodologies: "
            "years ≤%d use Global_2000_2020 (UN-adjusted, unconstrained, ppp); "
            "years ≥%d use Global_2015_2030 R2025A (constrained, 100m). "
            "Pre-2015 rasters will be reprojected onto the R2025A grid, but "
            "regional totals may show a step change at the 2014/2015 boundary.",
            start_year, end_year, _GLOBAL1_LAST_YEAR, _GLOBAL1_LAST_YEAR + 1,
        )

    reference_is_global2 = has_global2

    def _is_reference_version(year: int) -> bool:
        return (year > _GLOBAL1_LAST_YEAR) if reference_is_global2 else (year <= _GLOBAL1_LAST_YEAR)

    arrays: list[xr.DataArray] = []
    encoding: dict = {}
    reference: xr.DataArray | None = None
    for year, path in zip(years, files):
        da_year = xr.open_dataset(path)["pop_total"]
        da_year = da_year.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="x", y_dim="y")
        if not encoding:
            encoding = dict(da_year.encoding)
        if _is_reference_version(year) and reference is None:
            reference = da_year
        arrays.append(da_year)

    assert reference is not None
    aligned: list[xr.DataArray] = []
    for year, da_year in zip(years, arrays):
        if _is_reference_version(year):
            aligned.append(da_year)
        else:
            # odc.reproject renames spatial dims to latitude/longitude — rename
            # back to x/y and snap coord values to the reference so concat
            # treats them as the same grid.
            rep = reproject_population_to(da_year, reference)
            rep = rep.rename({d: t for d, t in {"longitude": "x", "latitude": "y"}.items() if d in rep.dims})
            rep = rep.assign_coords(x=reference.x, y=reference.y)
            aligned.append(rep)

    da = xr.concat(aligned, dim="time", coords="minimal", compat="override", join="override")
    da = da.fillna(0).astype("float32")
    da = da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="x", y_dim="y")

    da.name = "population"
    da.attrs.update(
        long_name="Population count per pixel",
        standard_name="number_concentration_of_people_in_air",
        units="people",
        source=f"WorldPop {start}..{end}",
    )
    da.encoding = encoding

    return da
