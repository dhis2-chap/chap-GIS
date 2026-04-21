"""Generic country-wide mosquito exposure surface (30 m), lazy xarray pipeline.

Driver script that composes chap_gis functions. Everything stays dask-lazy
until the terminal ``to_netcdf`` / ``to_raster`` calls, which trigger a
single combined compute.
"""

from __future__ import annotations

from pathlib import Path
import logging

import xarray as xr
from cyclopts import App

import chap_gis as cgis
from chap_gis.grid import reproject_to


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(module)s:%(funcName)s %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = App(name="exposure-analysis", help=__doc__)


@app.default
def run(
    *,
    country: str,
    year_worldcover: int = 2021,
    year_chelsa: int = 2021,
    year_worldpop: int = 2021,
    resolution_m: float = 30.0,
    out_dir: Path = Path("data/outputs"),
) -> None:
    """Compute and write the country-wide exposure rasters."""
    
    logger.info(f'Starting malaria exposure analysis for country: {country}')
    out_dir.mkdir(parents=True, exist_ok=True)


    ###########################################################################
    # analysis setup
    logger.info('\n\n##########################################################')
    logger.info('Setting up analysis')

    # load area of interest / country with buffer
    logger.info('Loading country geometry')
    aoi = cgis.io.boundaries.load(country, level=0)
    buffered = cgis.aoi.buffered(aoi, distance=0.0027)  # ~300 m buffer  # NOTE: note sure that we need...

    # build analysis grid for area
    logger.info(f'Building analysis grid at {resolution_m} meter resolution')
    grid = cgis.grid.build_grid(
        aoi, resolution=resolution_m / 111_000, crs="EPSG:4326"
    )


    ###########################################################################
    # prepare data sources
    logger.info('\n\n##########################################################')
    logger.info('Preparing data sources')


    ###############
    # landcover

    # load landcover and project to analysis grid
    logger.info(f'Loading worldcover data for year {year_worldcover}')
    landcover = (
        cgis.io.worldcover.load(buffered, year=year_worldcover)
        .pipe(reproject_to, grid, "mode")
        .astype("uint8")
    )


    ##############
    # elevation

    # load elevation and project to analysis grid
    logger.info('Loading elevation data')
    elev_native = cgis.io.elevation.load(buffered)
    elev = elev_native.pipe(reproject_to, grid, "bilinear")


    #############
    # temperature

    # load monthly temperature data and calculate annual mean
    logger.info(f'Loading and computing chelsa temperature data for year {year_chelsa}')
    tas_annual = (
        cgis.io.chelsa.load_monthly_tas(aoi, year=year_chelsa)
        .pipe(cgis.climate.annual_mean)
    )

    # reproject to analysis grid
    tas_on_grid = tas_annual.pipe(reproject_to, grid, "bilinear")


    #############
    # downscale temperature

    # create coarse elevation grid at same res as temperature
    logger.info(f'Coarsen elevation data to resolution of chelsa temperature data')
    coarse_elev_on_grid = (
        elev_native
        .pipe(reproject_to, tas_annual, "average")
        .pipe(reproject_to, grid, "bilinear")
    )

    # downscale temperature based on elevation
    logger.info('Combine temperature with elevation to downscale to analysis grid')
    temperature = cgis.climate.lapse_rate_downscale(
        tas_on_grid, coarse_elev_on_grid, elev
    )


    ################
    # population

    # load population
    logger.info('Loading worldpop population data')
    population = cgis.io.worldpop.load(
        country, year=year_worldpop
    )

    # reproject to analysis grid
    # NOTE: right now inflates the pop by repeating total counts
    # TODO: need to convert to divide the pop by new cell size division
    population = population.pipe(reproject_to, grid, 'nearest')


    ################
    # rice fields

    # load rice fields data and project to analysis grid
    logger.info('Loading rice data')
    rice_mask = (
        cgis.io.rice.load(country)
        .pipe(reproject_to, grid, "nearest")
        .pipe(lambda r: (r > 0).rio.write_crs(grid.rio.crs))  # TODO: dont think we need to write crs here, or should at least do so consistently
    )


    #############################################################################
    # analysis
    logger.info('\n\n############################################################')
    logger.info('Running analyses')


    ##############
    # analysis: compute suitability

    # calculate thermal malaria suitability from the downscaled temperature
    logger.info('Computing thermal suitability')
    suitability = temperature.pipe(cgis.suitability.thermal_suitability)


    #################
    # analysis: compute breeding sites

    # compute breeding site mask from landcover and rice fields
    logger.info('Compute breeding sites')
    breeding = cgis.landcover.breeding_site_mask(
        landcover, rice=rice_mask, water_edge_buffer=2
    )


    #################
    # analysis: compute exposure based on various layers

    # compute exposure layer based on breeding sites, elevation, suitability, land mask, and water mask
    logger.info('Compute exposure layer')
    expo = cgis.exposure.exposure(
        breeding,
        elev,
        suitability,
        pixel_m=resolution_m,
        land_mask=cgis.landcover.land_mask(landcover),
        water_mask=cgis.landcover.water_mask(landcover),
    )

    # weight exposure by population
    logger.info('Weight exposure by population')
    pop_exposure = (population * expo).rename("pop_exposure")
    pop_exposure.attrs.update(long_name="Population-weighted exposure", units="people")
    pop_exposure = pop_exposure.rio.write_crs(grid.rio.crs)


    ###################################################################################
    # finalizing
    logger.info('\n\n##################################################################')
    logger.info('Finalizing and outputting results')

    # create final grid with all layers
    logger.info('Creating final analysis dataset')
    out_ds = xr.Dataset(
        {
            "temperature": temperature,
            "suitability": suitability,
            "population": population,
            "exposure": expo,
            "pop_exposure": pop_exposure,
        }
    )

    # write to final output netcdf - all lazy steps get computed here
    logger.info(f'Outputting to {out_dir}')
    nc_path = out_dir / f"{country.lower()}_exposure.nc"
    out_ds.to_netcdf(nc_path)
    print(f"  Wrote {nc_path}")

    # analysis: compute population exposure hotspots
    # _, stats = cgis.hotspots.identify_hotspots(pop_exposure, population)
    # print(f"  Hotspot threshold (top 10%): {stats['threshold']:.3f}")
    # if "pct" in stats:
    #     print(f"  People in hotspots: {stats['hotspot_pop']:,.0f} ({stats['pct']:.1f}%)")


if __name__ == "__main__":
    app()
