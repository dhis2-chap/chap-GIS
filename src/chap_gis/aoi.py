"""Area-of-interest helpers."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box


def aoi_bounds(aoi: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) total bounds of the AOI."""
    return tuple(aoi.total_bounds)


def buffered(aoi: gpd.GeoDataFrame, distance: float) -> gpd.GeoDataFrame:
    """Return a copy of the AOI with geometry buffered by `distance` (CRS units)."""
    out = aoi.copy()
    out["geometry"] = aoi.geometry.buffer(distance)
    return out


def aoi_from_bbox(
    minx: float, miny: float, maxx: float, maxy: float, crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """Build a 1-row AOI GeoDataFrame from a bounding box."""
    return gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=crs)
