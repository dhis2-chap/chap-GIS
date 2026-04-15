"""Data-source loaders for chap_gis.

All loaders return xarray objects on the data source's **native grid** with
CRS written via ``rioxarray``. Reprojection to a target grid is the caller's
responsibility.
"""

from . import boundaries, cache, chelsa, elevation, worldcover, worldpop

__all__ = ["boundaries", "cache", "chelsa", "elevation", "worldcover", "worldpop"]
