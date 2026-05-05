"""Country-wide malaria exposure pipeline (pure xarray transformation).

Inputs are pre-loaded xarray DataArrays — the caller (typically the chap-gis
``analyze`` CLI) handles all I/O. Output is a lazy ``xr.Dataset`` with seven
layers: ``breeding``, ``elev``, ``temperature``, ``suitability``,
``population``, ``expo``, ``pop_exposure``.
"""

from __future__ import annotations

import logging

import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

import chap_gis as cgis
from chap_gis.grid import reproject_population_to, reproject_to


logger = logging.getLogger(__name__)


class MalariaExposureParams(BaseModel):
    """All tunable parameters for the malaria exposure pipeline."""

    model_config = ConfigDict(frozen=True)

    resolution_m: float = Field(default=30.0, gt=0)
    aoi_buffer_deg: float = Field(default=0.0027, ge=0)
    meters_per_degree: int = Field(default=111_000, gt=0)
    water_edge_buffer_pixels: int = Field(default=2, ge=0)


def run(
    *,
    aoi,
    landcover_native: xr.DataArray,
    elev_native: xr.DataArray,
    tas_monthly: xr.DataArray,
    population_native: xr.DataArray,
    rice_native: xr.DataArray,
    params: MalariaExposureParams = MalariaExposureParams(),
) -> xr.Dataset:
    """Compute the seven exposure layers as a single (lazy) ``xr.Dataset``.

    The seven outputs share upstream nodes (elevation, landcover, suitability),
    so callers should typically run ``.compute()`` once before persisting to
    de-duplicate that work versus computing each variable independently.
    """
    logger.info("Building analysis grid at %s m resolution", params.resolution_m)
    grid = cgis.grid.build_grid(
        aoi,
        resolution=params.resolution_m / params.meters_per_degree,
        crs="EPSG:4326",
    )

    logger.info("Reprojecting landcover, elevation, temperature, population, rice onto grid")
    landcover = landcover_native.pipe(reproject_to, grid, "mode").astype("uint8")
    elev = elev_native.pipe(reproject_to, grid, "bilinear")

    tas_annual = tas_monthly.pipe(cgis.climate.annual_mean)
    tas_on_grid = tas_annual.pipe(reproject_to, grid, "bilinear")
    coarse_elev_on_grid = (
        elev_native
        .pipe(reproject_to, tas_annual, "average")
        .pipe(reproject_to, grid, "bilinear")
    )
    temperature = cgis.climate.lapse_rate_downscale(
        tas_on_grid, coarse_elev_on_grid, elev
    )

    population = population_native.pipe(reproject_population_to, grid, "bilinear")
    rice_mask = (
        rice_native
        .pipe(reproject_to, grid, "nearest")
        .pipe(lambda r: (r > 0).rio.write_crs(grid.rio.crs))
    )

    logger.info("Computing thermal suitability, breeding sites, exposure")
    suitability = temperature.pipe(cgis.suitability.thermal_suitability)
    breeding = cgis.landcover.breeding_site_mask(
        landcover, rice=rice_mask, water_edge_buffer=params.water_edge_buffer_pixels
    )
    expo = cgis.exposure.exposure(
        breeding,
        elev,
        suitability,
        pixel_m=params.resolution_m,
        land_mask=cgis.landcover.land_mask(landcover),
        water_mask=cgis.landcover.water_mask(landcover),
    )

    pop_exposure = (population * expo).rename("pop_exposure")
    pop_exposure.attrs.update(long_name="Population-weighted exposure", units="people")
    pop_exposure = pop_exposure.rio.write_crs(grid.rio.crs)

    return xr.Dataset(
        {
            "breeding": breeding,
            "elev": elev,
            "temperature": temperature,
            "suitability": suitability,
            "population": population,
            "expo": expo,
            "pop_exposure": pop_exposure,
        }
    )
