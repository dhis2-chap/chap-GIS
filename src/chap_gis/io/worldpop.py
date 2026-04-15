"""WorldPop population loader + density-preserving reproject helper."""

from __future__ import annotations

from pathlib import Path

import rioxarray
import xarray as xr

from ..grid import reproject_to
from .cache import cache_dir, download_file

WORLDPOP_CONSTRAINED_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
    "{year}/maxar_v1/{iso3}/{iso3_lower}_ppp_{year}_constrained.tif"
)


def load(
    iso3: str,
    year: int = 2020,
    path: str | Path | None = None,
    chunks: str | dict = "auto",
) -> xr.DataArray:
    """Load a WorldPop people-per-pixel raster as a lazy DataArray."""
    if path is None:
        iso3 = iso3.upper()
        fname = f"{iso3.lower()}_ppp_{year}_constrained.tif"
        url = WORLDPOP_CONSTRAINED_URL.format(
            year=year, iso3=iso3, iso3_lower=iso3.lower()
        )
        path = download_file(url, cache_dir() / fname, label=f"WorldPop {iso3} {year}")

    da = rioxarray.open_rasterio(path, masked=True, chunks=chunks).squeeze(
        "band", drop=True
    )
    da = da.astype("float32")
    da = da.where(da >= 0)
    da.name = "population"
    da.attrs.update(
        long_name="Population count per pixel",
        units="people",
        source=f"WorldPop constrained {year}",
    )
    return da


def reproject_density(
    src: xr.DataArray, target: xr.DataArray, resampling: str = "bilinear"
) -> xr.DataArray:
    """Reproject a people-per-pixel raster onto `target`, preserving totals.

    WorldPop is stored as people per pixel, so reprojection alone changes
    totals whenever pixel areas differ. This helper rescales by the
    destination/source pixel-area ratio. Stays lazy via :func:`reproject_to`.
    """
    src_t = src.rio.transform()
    src_area = abs(src_t.a * src_t.e)
    dst_t = target.rio.transform()
    dst_area = abs(dst_t.a * dst_t.e)

    reprojected = reproject_to(src, target, resampling=resampling)
    out = reprojected * (dst_area / src_area)
    out = out.where(out >= 0)
    out.name = src.name
    out.attrs = {**src.attrs, "history": "reproject_density(target)"}
    return out.rio.write_crs(target.rio.crs)
