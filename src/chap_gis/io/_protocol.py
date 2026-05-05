"""``DataSource`` protocol that every loader in :mod:`chap_gis.io` satisfies.

The protocol is shaped after :mod:`dhis2eo.data` so that chap_gis loaders can
be slotted into the climate-api / dhis2eo pipeline with minimal glue. Loader
modules are duck-typed against this protocol — there is no base class and no
inheritance chain.

Post-conditions every conformant ``load`` guarantees:

* CRS is written to the returned DataArray (``rio.write_crs("EPSG:4326")``).
* Spatial dims are ``(y, x)``; if a time dimension is present it is named
  ``time``.
* ``DataArray.name`` is set.
* ``DataArray.attrs`` populates at least ``long_name``, ``standard_name``,
  ``units``, and ``source``.

Conformance is verified at runtime by ``isinstance(module, DataSource)``; see
``tests/test_protocol.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import xarray as xr
from geopandas import GeoDataFrame

from dhis2eo.utils.types import BBox, DateLike


@runtime_checkable
class DataSource(Protocol):
    """Structural protocol satisfied by every ``chap_gis.io.<source>`` module."""

    dataset_id: str
    """Stable identifier in the ``{provider}_{var}_{period}`` style used by
    climate-api YAML registries (e.g. ``worldpop_population_yearly``)."""

    def load(
        self,
        aoi: GeoDataFrame | None = None,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        country_code: str | None = None,
    ) -> xr.DataArray: ...

    def download(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        bbox: BBox | None = None,
        *,
        dirname: str | Path | None = None,
        prefix: str | None = None,
        country_code: str | None = None,
        overwrite: bool = False,
    ) -> list[Path]: ...
