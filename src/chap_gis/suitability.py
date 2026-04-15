"""Thermal suitability for mosquito-borne transmission."""

from __future__ import annotations

import numpy as np
import xarray as xr

T_OPTIMAL = 25.0
T_SIGMA = 5.0
T_MIN = 16.0
T_MAX = 34.0


def thermal_suitability(
    temperature: xr.DataArray,
    t_opt: float = T_OPTIMAL,
    sigma: float = T_SIGMA,
    t_min: float = T_MIN,
    t_max: float | None = T_MAX,
) -> xr.DataArray:
    """Mordecai/Villena Gaussian thermal performance curve for transmission.

    Returns a (0, 1]-valued DataArray with the same grid as `temperature`.
    Zero below `t_min` (and above `t_max`, if set). NaN where temperature is
    NaN.
    """
    s = np.exp(-(((temperature - t_opt) / sigma) ** 2))
    s = s.where(temperature >= t_min, 0.0)
    if t_max is not None:
        s = s.where(temperature <= t_max, 0.0)
    s = s.where(temperature.notnull())
    s.name = "suitability"
    s.attrs.update(
        long_name="Thermal suitability (Mordecai/Villena Gaussian TPC)",
        units="1",
        t_opt=t_opt,
        sigma=sigma,
        t_min=t_min,
        t_max=t_max if t_max is not None else "null",
    )
    s = s.rio.write_crs(temperature.rio.crs)
    return s
