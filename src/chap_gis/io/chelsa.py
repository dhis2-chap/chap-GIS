"""CHELSA monthly temperature loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

Scaling = Literal["auto", "kelvin_x10", "celsius_x10", "celsius"]


def _open_month(path: Path, chunks) -> xr.DataArray:
    return rioxarray.open_rasterio(path, masked=True, chunks=chunks).squeeze(
        "band", drop=True
    )


def _detect_scaling(da: xr.DataArray) -> str:
    """Probe a tiny window of the lazy array to decide CHELSA scaling."""
    probe = da.isel(y=slice(0, 64), x=slice(0, 64)).values
    sample = probe[np.isfinite(probe)]
    if sample.size == 0:
        return "celsius"
    m = float(np.mean(sample))
    if m > 2000:
        return "kelvin_x10"
    if m > 200:
        return "celsius_x10"
    return "celsius"


def load_monthly_tas(
    chelsa_dir: str | Path,
    year: int,
    filename_template: str = "CHELSA_tas_{month:02d}_{year}_V.2.1.tif",
    chunks: str | dict = "auto",
    scaling: Scaling = "auto",
) -> xr.DataArray:
    """Load 12 monthly CHELSA near-surface air temperature rasters.

    Returns a lazy DataArray with dims ``(time, y, x)`` in degrees Celsius.
    ``scaling`` selects the unit conversion; ``"auto"`` probes a tiny window
    (64x64) to decide between K×10 / °C×10 / °C.
    """
    chelsa_dir = Path(chelsa_dir)
    arrays = []
    for month in range(1, 13):
        path = chelsa_dir / filename_template.format(month=month, year=year)
        if not path.exists():
            raise FileNotFoundError(path)
        arrays.append(_open_month(path, chunks))

    da = xr.concat(arrays, dim="time").astype("float32")
    da = da.assign_coords(
        time=pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    )

    mode = _detect_scaling(da) if scaling == "auto" else scaling
    if mode == "kelvin_x10":
        da = da / 10.0 - 273.15
    elif mode == "celsius_x10":
        da = da / 10.0
    # else: celsius, no-op

    da.name = "tas"
    da.attrs.update(
        long_name="Near-surface air temperature (monthly mean)",
        standard_name="air_temperature",
        units="degC",
        source=f"CHELSA v2.1 monthly tas {year}",
        scaling=mode,
    )
    return da
