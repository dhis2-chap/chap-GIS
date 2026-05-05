"""``analyze`` subcommand: load inputs, run the malaria-exposure pipeline, write NetCDFs."""

from __future__ import annotations

import logging
from pathlib import Path

import chap_gis as cgis
from chap_gis.pipelines.malaria_exposure import MalariaExposureParams, run


logger = logging.getLogger(__name__)


OUTPUT_FILENAMES = {
    "breeding":     "breeding.nc",
    "elev":         "elevation.nc",
    "temperature":  "temperature.nc",
    "suitability":  "suitability.nc",
    "population":   "population.nc",
    "expo":         "exposure.nc",
    "pop_exposure": "pop_exposure.nc",
}


def analyze(
    *,
    country: str,
    year_worldcover: int = 2021,
    year_chelsa: int = 2021,
    year_worldpop: int = 2021,
    resolution_m: float = 30.0,
    out_dir: Path = Path("data/outputs"),
) -> None:
    """Compute and write the country-wide exposure rasters."""
    logger.info(f"Starting malaria exposure analysis for country: {country}")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = MalariaExposureParams(resolution_m=resolution_m)

    logger.info("Loading country geometry")
    aoi = cgis.io.boundaries.load(country, level=0)
    buffered_aoi = cgis.aoi.buffered(aoi, distance=params.aoi_buffer_deg)

    logger.info(
        f"Loading worldcover {year_worldcover}, elevation, CHELSA {year_chelsa}, "
        f"worldpop {year_worldpop}, rice"
    )
    landcover_native = cgis.io.worldcover.load(
        buffered_aoi,
        start=year_worldcover,
        end=year_worldcover,
        country_code=country,
    )
    elev_native = cgis.io.elevation.load(buffered_aoi, country_code=country)
    tas_monthly = cgis.io.chelsa.load(
        aoi,
        start=f"{year_chelsa}-01",
        end=f"{year_chelsa}-12",
        country_code=country,
    )
    population_native = cgis.io.worldpop.load(
        start=year_worldpop, end=year_worldpop, country_code=country
    )
    rice_native = cgis.io.rice.load(country_code=country)

    ds = run(
        aoi=aoi,
        landcover_native=landcover_native,
        elev_native=elev_native,
        tas_monthly=tas_monthly,
        population_native=population_native,
        rice_native=rice_native,
        params=params,
    ).compute()

    logger.info(f"Writing outputs to {out_dir}")
    for var_name, filename in OUTPUT_FILENAMES.items():
        ds[var_name].to_netcdf(out_dir / filename)
    logger.info("Finished!")
