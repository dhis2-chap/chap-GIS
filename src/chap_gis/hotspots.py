"""Identify exposure hotspots."""

from __future__ import annotations

import xarray as xr


def identify_hotspots(
    pop_exposure: xr.DataArray,
    population: xr.DataArray | None = None,
    percentile: float = 90,
) -> tuple[xr.DataArray, dict]:
    """Flag pixels whose population-weighted exposure exceeds a percentile.

    The hotspot mask itself is kept lazy (dask-backed). ``stats`` contains
    summary scalars computed via xarray reductions — these force a compute
    of the scalar reductions only.
    """
    positive = pop_exposure.where(pop_exposure > 0)
    threshold = positive.quantile(percentile / 100.0, skipna=True)
    hotspot = ((pop_exposure >= threshold) & pop_exposure.notnull()).rename("hotspot")

    stats: dict = {"threshold": float(threshold.values)}
    if population is not None:
        hotspot_pop = population.where(hotspot).sum(skipna=True)
        total_pop = population.where(population.notnull()).sum(skipna=True)
        hp = float(hotspot_pop.values)
        tp = float(total_pop.values)
        stats.update(
            hotspot_pop=hp,
            total_pop=tp,
            pct=(100.0 * hp / tp) if tp > 0 else 0.0,
        )
    return hotspot, stats
