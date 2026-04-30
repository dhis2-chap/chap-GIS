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
import xarray as xr
from cyclopts import App
from rasterio.enums import Resampling
import chap_gis as cgis
from shapely.geometry import box
from chap_gis.grid import reproject_to, reproject_population_to


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(module)s:%(funcName)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = App(name="exposure-analysis", help=__doc__)


def save_and_clear(da: xr.DataArray, name: str, path: Path) -> None:
    """Helper function to save a DataArray to disk and clear it from memory."""
    logger.info(f"Writing {name}...")
    da.rename(name).to_netcdf(path)
    del da
    gc.collect()


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

    # 1. Analysis Setup
    logger.info('\n' + '#' * 58 + '\nSetting up analysis')
    aoi = cgis.io.boundaries.load(country, level=0)
    
    # Grid creation (using meters to degrees approximation)
    grid = cgis.grid.build_grid(
        aoi, resolution=resolution_m / 111_000, crs="EPSG:4326"
    )
    logger.info(f"Grid built. Bounds: {grid.rio.bounds()} | CRS: {grid.rio.crs}")

    # Buffer for data fetching to avoid edge artifacts
    buffered = aoi.to_crs("EPSG:3857").buffer(300).to_crs("EPSG:4326")

    # 2. Data Preparation
    logger.info('\n' + '#' * 58 + '\nPreparing data sources')

    # --- Landcover ---
    logger.info(f'Loading WorldCover {year_worldcover}')
    landcover = (
        cgis.io.worldcover.load(buffered, year=year_worldcover,country=country)
        .rio.reproject_match(grid, resampling=Resampling.mode)
        .fillna(0)
        .astype("uint8")
    )

    # --- Elevation ---
    logger.info('Loading Elevation')
    elev_native = cgis.io.elevation.load(buffered, country=country)
    
    # Ensure overlap
    if not box(*elev_native.rio.bounds()).intersects(box(*grid.rio.bounds())):
        logger.warning("Elevation extent mismatch; refetching with larger buffer...")
        buffered_large = aoi.to_crs("EPSG:3857").buffer(1000).to_crs("EPSG:4326")
        elev_native = cgis.io.elevation.load(buffered_large, country=country)

    elev = elev_native.rio.reproject_match(grid, resampling=Resampling.bilinear)

    # --- Temperature (CHELSA) ---
    logger.info(f'Loading CHELSA Temperature {year_chelsa}')
    tas_annual = (
        cgis.io.chelsa.load_monthly_tas(aoi, year=year_chelsa, country=country)
        .pipe(cgis.climate.annual_mean)
    )

    # CRITICAL: Standardize dimension names to x/y before reprojection
    dim_map = {'longitude': 'x', 'latitude': 'y'}
    tas_annual = tas_annual.rename({k: v for k, v in dim_map.items() if k in tas_annual.dims})
    
    tas_on_grid = tas_annual.rio.reproject_match(grid, resampling=Resampling.bilinear)

    # --- Downscaling setup ---
    logger.info('Coarsening elevation for lapse-rate downscaling')
    # Coarsen elevation to match CHELSA resolution, then match back to grid
    coarse_elev_on_grid = (
        elev_native
        .rio.reproject_match(tas_annual, resampling=Resampling.average)
        .rename({k: v for k, v in dim_map.items() if k in elev_native.dims}) # ensure x,y
        .rio.reproject_match(grid, resampling=Resampling.bilinear)
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
        population.rename({k: v for k, v in dim_map.items() if k in population.dims})
        .rio.reproject_match(grid, resampling=Resampling.bilinear)
    )

    # --- Rice Mask ---
    logger.info('Loading Rice data')
    rice_mask = (
        cgis.io.rice.load(country)
        .rio.reproject_match(grid, resampling=Resampling.nearest)
        .pipe(lambda r: (r > 0).rio.write_crs(grid.rio.crs))
    )

    # 3. Exposure Analysis
    logger.info('\n' + '#' * 58 + '\nRunning analyses')

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

    # 4. Finalization
    logger.info('\n' + '#' * 58 + '\nFinalizing and outputting results')
    
    ds = xr.Dataset({
        "elev": elev,
        "breeding": breeding.astype(float),
        "temperature": temperature,
        "suitability": suitability,
        "population": population,
        "expo": expo,
        "pop_exposure": pop_exposure
    }).compute()

    for var_name in ds.data_vars:
        if var_name != 'spatial_ref':
            out_path = out_dir / f"{var_name}.nc"
            ds[var_name].to_netcdf(out_path)
            logger.info(f"Saved {var_name} to {out_path}")

    logger.info("Finished successfully!")

@app.command
def visualize(out_dir):
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
        fig.savefig(out_dir / f'{path.name}.png', dpi=300)
        
        # clear figure for next map
        plt.clf()

if __name__ == "__main__":
    app()
