"""Generic country-wide mosquito exposure surface (30 m), lazy xarray pipeline.

Driver script that composes chap_gis functions. Everything stays dask-lazy
until the terminal ``to_netcdf`` / ``to_raster`` calls, which trigger a
single combined compute.
"""

from __future__ import annotations

import os
from pathlib import Path
import logging
import gc

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray as xr
from cyclopts import App

from dhis2eo.integrations.chap import dataframe_to_chap_csv

import chap_gis as cgis
from chap_gis.grid import reproject_to, reproject_population_to
from chap_gis.aggregate import aggregate_to_regions


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(module)s:%(funcName)s %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = App(name="exposure-analysis", help=__doc__)


@app.command
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
    logger.info(grid)


    ###########################################################################
    # prepare data sources
    logger.info('\n\n##########################################################')
    logger.info('Preparing data sources')


    ###############
    # landcover

    # load landcover and project to analysis grid
    logger.info(f'Loading worldcover data for year {year_worldcover}')
    landcover = (
        cgis.io.worldcover.load(buffered, year=year_worldcover, country=country)
        .pipe(reproject_to, grid, "mode")
        .astype("uint8")
    )


    ##############
    # elevation

    # load elevation and project to analysis grid
    logger.info('Loading elevation data')
    elev_native = cgis.io.elevation.load(buffered, country=country)
    elev = elev_native.pipe(reproject_to, grid, "bilinear")


    #############
    # temperature

    # load monthly temperature data and calculate annual mean
    logger.info(f'Loading and computing chelsa temperature data for year {year_chelsa}')
    tas_annual = (
        cgis.io.chelsa.load_monthly_tas(aoi, year=year_chelsa, country=country)
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
    population = population.pipe(reproject_population_to, grid, 'bilinear')


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

    # Bundle layers into a single Dataset and trigger one combined compute.
    # The seven outputs share common upstream nodes (elevation, landcover,
    # suitability, ...), so a single .compute() de-duplicates that work vs
    # calling .to_netcdf() seven times on independent graphs.
    logger.info('Computing all layers in a single pass')
    ds = xr.Dataset({
        "breeding": breeding,
        "elev": elev,
        "temperature": temperature,
        "suitability": suitability,
        "population": population,
        "expo": expo,
        "pop_exposure": pop_exposure,
    }).compute()

    logger.info(f'Outputting to {out_dir}')
    output_filenames = {
        "breeding": "breeding.nc",
        "elev": "elevation.nc",
        "temperature": "temperature.nc",
        "suitability": "suitability.nc",
        "population": "population.nc",
        "expo": "exposure.nc",
        "pop_exposure": "pop_exposure.nc",
    }
    for var_name, filename in output_filenames.items():
        ds[var_name].to_netcdf(out_dir / filename)

    logger.info("Finished!")


@app.command
def visualize(
    out_dir: str,
) -> None:
    """Visualize the various dataset outputs from the analysis."""
    # open dataset
    out_dir = Path(out_dir).resolve()
    logger.info(f'Visualizing nc files in folder {out_dir}')

    import matplotlib.pyplot as plt

    # make maps for each nc file
    for path in out_dir.glob('*.nc'):
        logger.info('----------------------------------------------------------')
        logger.info(f'File: {path}')
        ds = xr.open_dataset(path)
        logger.info(ds)

        # prep data
        var = [v for v in ds.data_vars if v != 'spatial_ref'][0]
        logger.info(f'Prepping data for {var}')
        da = ds[var].coarsen(longitude=10, latitude=10, boundary="trim").mean()

        # plot and save
        logger.info('Plotting data')
        ax = plt.subplot()
        da.plot(ax=ax)
        fig = ax.get_figure()
        fig.savefig(out_dir / f'{path.stem}.png', dpi=300)
        
        # clear figure for next map
        plt.clf()


def _simulate_monthly_disease_data(regions_df, location_id_field):
    # set location id field
    regions_df['location_id'] = regions_df[location_id_field]
    regions_df = regions_df[['location_id']]
    regions_df = regions_df.drop_duplicates(subset=["location_id"])

    # create df for timeframe (hardcoded for now)
    time_df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=12*3, freq="MS")})

    # crossjoin regions with time
    final = regions_df.merge(time_df, how='cross')

    # add random disease data
    from random import uniform
    final['disease'] = [uniform(0, 30) for _ in range(len(final))]

    return final


@app.command
def aggregate(
    out_dir: str,
    country: str,
    level: int,
) -> None:
    """Aggregate the various dataset outputs to country administrative boundaries and output to chap CSV.
    
    Administrative boundaries are fetched dynamically from GeoBoundaries (should be easy later to switch out with custom geojson file).
    Disease data needs to exist as a CSV file in the inputs folder, with the name "disease_data_<countrycode>". 
    """
    out_dir = Path(out_dir).resolve()
    logger.info(f'Aggregating nc files in folder {out_dir} to country boundaries {country}-ADM{level}')

    # load admin boundaries for country and level
    logger.info('Loading boundary regions')
    gdf = cgis.io.boundaries.load(country, level=level)
    logger.info(gdf)

    # load disease data based on required location and naming convention
    logger.info('Loading disease data')
    filename = f'disease_data_{country.lower()}.csv'
    disease_path = out_dir.parent / 'inputs' / filename
    if disease_path.exists():
        logger.info(f'--> {disease_path}')
        df = pd.read_csv(disease_path, parse_dates=["time"])
    else:
        logger.info('--> Generating dummy data')
        df = _simulate_monthly_disease_data(regions_df=gdf, location_id_field='shapeName')

    assert 'location_id' in df.columns
    assert 'time' in df.columns
    assert 'disease' in df.columns

    # convert to xarray
    logger.info(df)
    output = df.set_index(["location_id", "time"]).to_xarray()
    logger.info(output)

    # aggregate for each nc file and join to a single output dataset
    for path in out_dir.glob('*.nc'):
        logger.info('')
        logger.info('----------------------------------------------------------')
        logger.info(f'File: {path}')

        # open nc file
        ds = rioxarray.open_rasterio(path)
        ds_name = path.stem

        # hacky fixes
        # TODO: we need to standardize this earlier in the pipelines and not cleanup here
        # squeeze away unneeded dimensions
        ds = ds.squeeze(drop=True)
        # add time dim
        ds = ds.expand_dims(time=[pd.Timestamp("2021-01-01")])

        # inspect input
        logger.info('Input xarray:')
        logger.info(ds)

        # aggregate to geojson
        # TODO: right now we have no way to map each nc file to specific statistic
        agg = aggregate_to_regions(ds, gdf, statistic='mean', id_field='location_id')

        # prefix "mean" variable with variable name before merging
        agg = agg.rename({'mean': f'{ds_name}_mean'})
        logger.info(agg)

        # cleanup for memory
        del ds
        gc.collect()

        # left join to main disease dataset
        output = xr.merge([output, agg], join='left')

    # prepare and output to chap csv
    # example of preparing data for chap, see: https://climate-tools.dhis2.org/guides/import-chap/harmonize-to-chap/
    logger.info(f'Merged dataset of all aggregates:')
    logger.info(output)
    out_path = out_dir / 'chap-output.csv'
    # map columns to chap names
    # these are the required columns, all others will be included with their original variables names
    column_map = {
        "time_period": "time",
        "location": "location_id",
        "disease_cases": "disease",
        "population": "population",
    }
    # output chap-csv
    dataframe_to_chap_csv(
        output.to_dataframe().reset_index(),
        column_map=column_map,
        freq='monthly',
        start=str(output['time'].min())[:7],  # these dont quite work yet, need to look up the expected format :P
        end=str(output['time'].max())[:7],  # these dont quite work yet, need to look up the expected format :P
        output_path=out_path,
    )


if __name__ == "__main__":
    app()
