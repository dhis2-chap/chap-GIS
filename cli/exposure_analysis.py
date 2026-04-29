"""Generic country-wide mosquito exposure surface (30 m), lazy xarray pipeline.

Driver script that composes chap_gis functions. Everything stays dask-lazy
until the terminal ``to_netcdf`` / ``to_raster`` calls, which trigger a
single combined compute.
"""

from __future__ import annotations

import gc
from pathlib import Path
import logging

import numpy as np
import geopandas as gpd
import rioxarray
import xarray as xr
from cyclopts import App

from dhis2eo.integrations.chap import dataframe_to_chap_csv

import chap_gis as cgis
from shapely.geometry import box
from chap_gis.grid import reproject_to, reproject_population_to
from chap_gis.aggregate import aggregate_to_regions


# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(module)s:%(funcName)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



# create CLI app
app = App(name="exposure-analysis", help=__doc__)



def save_and_clear(da: xr.DataArray, name: str, path: Path) -> None:
    """Helper function to save a DataArray to disk and clear it from memory."""
    logger.info(f"Writing {name}...")
    da.rename(name).to_netcdf(path)
    del da
    gc.collect()


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


    #####################
    # 1. Analysis Setup
    logger.info('\n\n' + '#' * 58 + '\nSetting up analysis')
    aoi = cgis.io.boundaries.load(country, level=0)
    
    # Grid creation (using meters to degrees approximation)
    grid = cgis.grid.build_grid(
        aoi, resolution=resolution_m / 111_000, crs="EPSG:4326"
    )
    logger.info(f"Grid built. Bounds: {grid.rio.bounds()} | CRS: {grid.rio.crs}")

    # Buffer for data fetching to avoid edge artifacts
    buffered = aoi.to_crs("EPSG:3857").buffer(300).to_crs("EPSG:4326")


    #######################
    # 2. Data Preparation
    logger.info('\n\n' + '#' * 58 + '\nPreparing data sources')

    # --- Landcover ---
    logger.info(f'Loading WorldCover {year_worldcover}')
    landcover = (
        cgis.io.worldcover.load(buffered, year=year_worldcover)
        .pipe(reproject_to, grid, "mode")
        .fillna(0)
        .astype("uint8")
    )

    # --- Elevation ---
    logger.info('Loading Elevation')
    elev_native = cgis.io.elevation.load(buffered)
    
    # Ensure overlap
    if not box(*elev_native.rio.bounds()).intersects(box(*grid.rio.bounds())):
        logger.warning("Elevation extent mismatch; refetching with larger buffer...")
        buffered_large = aoi.to_crs("EPSG:3857").buffer(1000).to_crs("EPSG:4326")
        elev_native = cgis.io.elevation.load(buffered_large)

    elev = elev_native.pipe(reproject_to, grid, "bilinear")

    # --- Temperature (CHELSA) ---
    logger.info(f'Loading CHELSA Temperature {year_chelsa}')
    tas_annual = (
        cgis.io.chelsa.load_monthly_tas(aoi, year=year_chelsa)
        .pipe(cgis.climate.annual_mean)
    )
    
    tas_on_grid = tas_annual.pipe(reproject_to, grid, "bilinear")

    # --- Downscaling setup ---
    logger.info('Coarsening elevation for lapse-rate downscaling')
    # Coarsen elevation to match CHELSA resolution, then match back to grid
    coarse_elev_on_grid = (
        elev_native
        .pipe(reproject_to, tas_annual, "average")
        .pipe(reproject_to, grid, "bilinear")
    )

    logger.info('Applying lapse-rate downscaling')
    temperature = cgis.climate.lapse_rate_downscale(
        tas_on_grid, coarse_elev_on_grid, elev
    )

    # --- Population ---
    logger.info('Loading WorldPop')
    population = cgis.io.worldpop.load(country, year=year_worldpop)
    if population.ndim > 2:
        population = population.squeeze(drop=True)
    
    population = (
        population
        .pipe(reproject_population_to, grid, 'bilinear')
    )

    # --- Rice Mask ---
    logger.info('Loading Rice data')
    rice_mask = (
        cgis.io.rice.load(country)
        .pipe(reproject_to, grid, "nearest")
        .pipe(lambda r: (r > 0).rio.write_crs(grid.rio.crs))  # TODO: dont think we need to write crs here, or should at least do so consistently
    )


    ########################
    # 3. Exposure Analysis
    logger.info('\n\n' + '#' * 58 + '\nRunning analyses')

    suitability = temperature.pipe(cgis.suitability.thermal_suitability)
    
    breeding = cgis.landcover.breeding_site_mask(
        landcover, rice=rice_mask, water_edge_buffer=2
    )

    expo = cgis.exposure.exposure(
        breeding,
        elev,
        suitability,
        pixel_m=resolution_m,
        land_mask=cgis.landcover.land_mask(landcover),
        water_mask=cgis.landcover.water_mask(landcover),
    )

    pop_exposure = (population * expo).rename("pop_exposure")
    pop_exposure.attrs.update(long_name="Population-weighted exposure", units="people")


    ###################
    # 4. Finalization
    logger.info('\n' + '#' * 58 + '\nFinalizing and outputting results')
    
    output_mapping = {
        "elev": elev,
        "breeding": breeding, #.astype(float),
        "temperature": temperature,
        "suitability": suitability,
        "population": population,
        "expo": expo,
        "pop_exposure": pop_exposure,
    }

    for da_name, da in output_mapping.items():
        out_path = out_dir / f"{da_name}.nc"
        logger.info(f"Saving {da_name} to {out_path}...")
        da.to_netcdf(out_path)  # this will automatically compute all lazy operations

    logger.info("Finished successfully!")


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
        da = ds[var].coarsen(x=10, y=10, boundary="trim").mean()

        # plot and save
        logger.info('Plotting data')
        ax = plt.subplot()
        da.plot(ax=ax)
        fig = ax.get_figure()
        fig.savefig(out_dir / f'{path.stem}.png', dpi=300)
        
        # clear figure for next map
        plt.clf()


@app.command
def aggregate(
    out_dir: str,
    geojson_file: str,
    id_field: str,
    #disease_csv: str,
) -> None:
    """Aggregate the various dataset outputs to a geojson file and output to chap CSV."""
    out_dir = Path(out_dir).resolve()
    logger.info(f'Aggregating nc files in folder {out_dir} to geojson file {geojson_file}')

    # open geojson file
    gdf = gpd.read_file(geojson_file)
    logger.info('Input geojson:')
    logger.info(gdf)

    # aggregate for each nc file
    for path in out_dir.glob('*.nc'):
        logger.info('')
        logger.info('----------------------------------------------------------')
        logger.info(f'File: {path}')

        # open nc file
        ds = rioxarray.open_rasterio(path)
        logger.info('Input xarray:')
        logger.info(ds)

        # aggregate to geojson
        # TODO: right now we have no way to map each nc file to specific statistic
        agg = aggregate_to_regions(ds, gdf, statistic='mean', id_field=id_field)

        # convert to df
        df = agg.to_dataframe()
        logger.info('Aggregated results:')
        # hacky add time column
        # TODO: all output nc files need to add `time`` dimension
        df['time'] = np.datetime64('2021-01-01')
        logger.info(df)

        # determine start and end from time column values
        start = df['time'].min()
        end = df['time'].max()
        logger.info(f'Time range detected: {start} to {end}')

        # write to chap compatible csv
        file_name = path.stem
        out_path = out_dir / f'{file_name}.csv'
        logger.info(f'Writing to CSV: {out_path}')
        df.to_csv(out_path)

    # finally, merge all csv files to a common region and time grid, including an input health csv dataset
    # see: https://climate-tools.dhis2.org/guides/import-chap/harmonize-to-chap/
    # ... 

    # output to chap csv
    #dataframe_to_chap_csv(...)


if __name__ == "__main__":
    app()
