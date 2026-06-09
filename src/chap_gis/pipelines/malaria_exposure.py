"""Country-wide malaria exposure pipeline (pure xarray transformation).

Inputs are pre-loaded xarray DataArrays — the caller (typically the chap-gis
``analyze`` CLI) handles all I/O. Output is a lazy ``xr.Dataset`` with seven
layers: ``breeding``, ``elev``, ``temperature``, ``suitability``,
``population``, ``expo``, ``pop_exposure``.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

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

    # Exposure kernel — distance/elevation decay (see chap_gis.exposure).
    horizontal_lambda_m: float = Field(default=651.0, gt=0)
    vertical_gamma_m: float = Field(default=22.5, gt=0)

    # Thermal suitability curve (see chap_gis.suitability).
    t_opt: float = Field(default=25.0)
    t_sigma: float = Field(default=5.0, gt=0)
    t_min: float = Field(default=16.0)
    t_max: float | None = Field(default=34.0)


class ThermalParams(BaseModel):
    """One thermal-suitability curve, as swept in :class:`ExposureSweepSpec`."""

    model_config = ConfigDict(frozen=True)

    t_opt: float = 25.0
    sigma: float = Field(default=5.0, gt=0)
    t_min: float = 16.0
    t_max: float | None = 34.0


class ExposureCombo(BaseModel):
    """A single resolved point of the exposure parameter grid."""

    model_config = ConfigDict(frozen=True)

    tag: str
    lambda_m: float
    gamma_m: float
    water_edge_buffer_pixels: int
    thermal: ThermalParams


def combo_tag(idx: int) -> str:
    """Stable column tag for the ``idx``-th combo in :meth:`ExposureSweepSpec.combos`."""
    return f"expo_{idx:03d}"


class ExposureSweepSpec(BaseModel):
    """A grid of exposure parameters to evaluate in one pass.

    The Cartesian product is enumerated by :meth:`combos` in a fixed nested
    order (water buffer → thermal → lambda → gamma) that the orchestration
    relies on to reuse the expensive distance transform: the breeding mask /
    distance field only changes with ``water_edge_buffer_pixels``, thermal
    suitability only with ``thermal``, and ``lambda_m``/``gamma_m`` are a cheap
    final kernel.
    """

    model_config = ConfigDict(frozen=True)

    lambda_m: list[float] = Field(default_factory=lambda: [651.0])
    gamma_m: list[float] = Field(default_factory=lambda: [22.5])
    water_edge_buffer_pixels: list[int] = Field(default_factory=lambda: [2])
    thermal: list[ThermalParams] = Field(default_factory=lambda: [ThermalParams()])
    base: MalariaExposureParams = Field(default_factory=MalariaExposureParams)

    def combos(self) -> list[ExposureCombo]:
        nt, nl, ng = len(self.thermal), len(self.lambda_m), len(self.gamma_m)
        out: list[ExposureCombo] = []
        for wb_i, wb in enumerate(self.water_edge_buffer_pixels):
            for th_i, th in enumerate(self.thermal):
                for lam_i, lam in enumerate(self.lambda_m):
                    for gam_i, gam in enumerate(self.gamma_m):
                        idx = ((wb_i * nt + th_i) * nl + lam_i) * ng + gam_i
                        out.append(
                            ExposureCombo(
                                tag=combo_tag(idx),
                                lambda_m=lam,
                                gamma_m=gam,
                                water_edge_buffer_pixels=wb,
                                thermal=th,
                            )
                        )
        return out


class ReprojectedLayers(NamedTuple):
    """Grid-aligned inputs shared by every combo for a single year."""

    grid: xr.DataArray
    landcover: xr.DataArray
    elev: xr.DataArray
    temperature: xr.DataArray
    population: xr.DataArray
    rice_mask: xr.DataArray


def reproject_layers(
    *,
    aoi,
    landcover_native: xr.DataArray,
    elev_native: xr.DataArray,
    tas_monthly: xr.DataArray,
    population_native: xr.DataArray,
    rice_native: xr.DataArray,
    params: MalariaExposureParams = MalariaExposureParams(),
) -> ReprojectedLayers:
    """Build the analysis grid and reproject every input onto it (lazy).

    These layers depend only on the year's raw inputs and ``params`` that
    affect geometry/resolution — not on the exposure kernel or thermal curve —
    so a parameter sweep computes them once per year and reuses them.
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
        .pipe(reproject_to, grid, "average")
        .pipe(lambda r: (r > 0).rio.write_crs(grid.rio.crs))
    )

    return ReprojectedLayers(grid, landcover, elev, temperature, population, rice_mask)


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
    layers = reproject_layers(
        aoi=aoi,
        landcover_native=landcover_native,
        elev_native=elev_native,
        tas_monthly=tas_monthly,
        population_native=population_native,
        rice_native=rice_native,
        params=params,
    )
    grid = layers.grid
    landcover, elev, temperature = layers.landcover, layers.elev, layers.temperature
    population, rice_mask = layers.population, layers.rice_mask

    logger.info("Computing thermal suitability, breeding sites, exposure")
    suitability = cgis.suitability.thermal_suitability(
        temperature,
        t_opt=params.t_opt,
        sigma=params.t_sigma,
        t_min=params.t_min,
        t_max=params.t_max,
    )
    breeding = cgis.landcover.breeding_site_mask(
        landcover, rice=rice_mask, water_edge_buffer=params.water_edge_buffer_pixels
    )
    expo = cgis.exposure.exposure(
        breeding,
        elev,
        suitability,
        pixel_m=params.resolution_m,
        horizontal_lambda_m=params.horizontal_lambda_m,
        vertical_gamma_m=params.vertical_gamma_m,
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
