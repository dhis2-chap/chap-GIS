"""geoBoundaries ADM loader."""

from __future__ import annotations

import geopandas as gpd

from .cache import cache_dir

GEOBOUNDARIES_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
    "releaseData/gbOpen/{iso3}/ADM{level}/geoBoundaries-{iso3}-ADM{level}.geojson"
)


def load(iso3: str, level: int = 0, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Load an administrative boundary from geoBoundaries.

    The GeoJSON is cached on disk under ``cache_dir()``. If the cached file is
    a DHIS2 organisation-unit export (columns ``id``/``name``/``code``...) it is
    normalised to geoBoundaries column names (``shapeID``/``shapeName``) so
    downstream code can treat both sources uniformly.
    """
    iso3 = iso3.upper()
    cache = cache_dir() / f"geoBoundaries-{iso3}-ADM{level}.geojson"
    if cache.exists():
        gdf = gpd.read_file(cache)
    else:
        url = GEOBOUNDARIES_URL.format(iso3=iso3, level=level)
        gdf = gpd.read_file(url)
        gdf.to_file(cache, driver="GeoJSON")

    if "shapeName" not in gdf.columns and {"id", "name"}.issubset(gdf.columns):
        gdf = gdf.rename(columns={"id": "shapeID", "name": "shapeName"})

    return gdf.to_crs(crs)
