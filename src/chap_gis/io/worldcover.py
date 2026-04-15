"""ESA WorldCover 10 m land-cover loader."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from .cache import cache_dir, download_file

WC_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
WC_TILE_SIZE_DEG = 3


def tile_name(lat_sw: int, lon_sw: int) -> str:
    """Encode a WorldCover tile name from its SW-corner lat/lon (integer degrees)."""
    ns = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{-lat_sw:02d}"
    ew = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{-lon_sw:03d}"
    return f"{ns}{ew}"


def tiles_for_bounds(bounds: tuple[float, float, float, float]) -> list[str]:
    """Return the list of WorldCover tile names covering a geographic bbox."""
    minx, miny, maxx, maxy = bounds
    lat_start = math.floor(miny / WC_TILE_SIZE_DEG) * WC_TILE_SIZE_DEG
    lat_stop = math.floor((maxy - 1e-9) / WC_TILE_SIZE_DEG) * WC_TILE_SIZE_DEG
    lon_start = math.floor(minx / WC_TILE_SIZE_DEG) * WC_TILE_SIZE_DEG
    lon_stop = math.floor((maxx - 1e-9) / WC_TILE_SIZE_DEG) * WC_TILE_SIZE_DEG
    tiles = []
    for lat in range(lat_start, lat_stop + 1, WC_TILE_SIZE_DEG):
        for lon in range(lon_start, lon_stop + 1, WC_TILE_SIZE_DEG):
            tiles.append(tile_name(lat, lon))
    return tiles


def _download_tile(tile: str, year: int = 2021) -> Path:
    filename = f"ESA_WorldCover_10m_{year}_v200_{tile}_Map.tif"
    url = f"{WC_BASE}/{filename}"
    return download_file(url, cache_dir() / filename, label=f"WorldCover {tile}")


def load(
    aoi: gpd.GeoDataFrame,
    year: int = 2021,
    chunks: str | dict = "auto",
) -> xr.DataArray:
    """Load WorldCover tiles covering `aoi` as a lazy mosaic DataArray.

    Each tile is opened lazily via :func:`rioxarray.open_rasterio`; the
    mosaic is assembled with :func:`xarray.combine_by_coords` so the whole
    returned array is dask-backed. Callers reproject/aggregate to the target
    grid (see :func:`chap_gis.grid.reproject_to`).
    """
    if str(aoi.crs) != "EPSG:4326":
        aoi = aoi.to_crs("EPSG:4326")
    bounds = tuple(aoi.total_bounds)

    tiles = [
        rioxarray.open_rasterio(
            _download_tile(t, year=year), masked=False, chunks=chunks
        ).squeeze("band", drop=True)
        for t in tiles_for_bounds(bounds)
    ]
    mosaic = xr.combine_by_coords(tiles, combine_attrs="override")
    if isinstance(mosaic, xr.Dataset):
        (name,) = mosaic.data_vars
        mosaic = mosaic[name]

    minx, miny, maxx, maxy = bounds
    mosaic = mosaic.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    mosaic.name = "landcover"
    mosaic.attrs.update(
        long_name="ESA WorldCover land cover class",
        standard_name="land_cover_lccs_class",
        units="1",
        source=f"ESA WorldCover v200 {year}",
    )
    return mosaic.rio.write_crs("EPSG:4326")
