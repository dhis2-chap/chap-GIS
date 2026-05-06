"""chap-gis CLI: country-wide mosquito exposure pipeline.

Subcommands compose ``chap_gis`` library functions. The ``analyze`` step keeps
everything dask-lazy until a single combined ``.compute()``; ``visualize`` and
``aggregate`` operate on the resulting NetCDF files.
"""

from __future__ import annotations

import logging

from cyclopts import App
from xarray import merge

from .analyze import analyze
from .visualize import visualize
from .aggregate import aggregate
from .merge import merge


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(module)s:%(funcName)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


app = App(name="chap-gis", help=__doc__)
app.command(analyze)
app.command(visualize)
app.command(aggregate)
app.command(merge)


__all__ = ["app"]
