"""Data-source loaders for chap_gis.

All loaders return xarray objects on the data source's **native grid** with
CRS written via ``rioxarray``. Reprojection to a target grid is the caller's
responsibility.

Each ``chap_gis.io.<source>`` module satisfies the :class:`DataSource`
protocol and registers itself in :data:`SOURCES` under its ``dataset_id``.
"""

from . import boundaries, cache, chelsa, crop, elevation, rice, worldcover, worldpop
from ._protocol import DataSource


SOURCES: dict[str, DataSource] = {
    chelsa.dataset_id: chelsa,
    crop.dataset_id: crop,
    elevation.dataset_id: elevation,
    rice.dataset_id: rice,
    worldcover.dataset_id: worldcover,
    worldpop.dataset_id: worldpop,
}


__all__ = [
    "DataSource",
    "SOURCES",
    "boundaries",
    "cache",
    "chelsa",
    "crop",
    "elevation",
    "rice",
    "worldcover",
    "worldpop",
]
