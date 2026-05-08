from pathlib import Path
from geopandas import GeoDataFrame
import pandas as pd
from typing import Optional
import xarray as xr
import numpy as np
import chap_gis as cgis

import logging

logger = logging.getLogger(__name__)
from chap_gis.aggregate import aggregate_population_by_year, aggregate_temperature_by_month, aggregate_to_regions
from chap_gis.pipelines.malaria_exposure import run as run_exposure_pipeline
from chap_gis.pipelines.malaria_exposure import MalariaExposureParams

def _simulate_monthly_disease_data(regions_df: pd.DataFrame, location_id_field: str) -> pd.DataFrame:
    """
    Generates dummy disease data for the given regions.
    """
    df = regions_df[[location_id_field]].copy()
    df = df.rename(columns={location_id_field: 'location_id'})
    df = df.drop_duplicates(subset=["location_id"])

    # Create 3 years of monthly timestamps
    time_df = pd.DataFrame({"time": pd.date_range("2017-01-01", periods=36, freq="MS")})

    # Cross join to create (location x time) grid
    final = df.merge(time_df, how='cross')
    final['disease'] = np.random.uniform(0, 30, size=len(final))

    return final

def _calculate_monthly_exposure_from_vars(
    aoi: GeoDataFrame,
    gdf: GeoDataFrame,
    pop_native: xr.DataArray,
    tas_native: xr.DataArray,
    country_code: str,
    params: MalariaExposureParams
) -> xr.Dataset:
    """
    Calculates monthly exposure using already loaded population and temperature rasters.
    """
    logger.info("Running monthly exposure pipeline on loaded rasters...")

    # 1. Load the remaining static layers needed for the exposure model
    buffered_aoi = cgis.aoi.buffered(aoi, distance=params.aoi_buffer_deg)

    requested_year = int(tas_native.time.dt.year.max())
    worldcover_year = max(2020, min(requested_year, 2021)) 
    
    logger.info(f"Requested year {requested_year} for LandCover. Using available year: {worldcover_year}")

    landcover_native = cgis.io.worldcover.load(
        buffered_aoi, 
        start=worldcover_year,  # Use the clamped year
        end=worldcover_year,
        country_code=country_code
    )
    elev_native = cgis.io.elevation.load(buffered_aoi, country_code=country_code)
    rice_native = cgis.io.rice.load(country_code=country_code)

    # 2. Run the pixel-level pipeline
    # This automatically broadcasts the static layers across the monthly 'tas_native'
    ds_pixel = run_exposure_pipeline(
        aoi=aoi,
        landcover_native=landcover_native,
        elev_native=elev_native,
        tas_monthly=tas_native,
        population_native=pop_native,
        rice_native=rice_native,
        params=params,
    ).compute()

    # 3. Aggregate to regions
    logger.info("Aggregating 'pop_exposure' to administrative regions...")
    expo_agg_ds = aggregate_to_regions(
        ds_pixel["pop_exposure"], 
        gdf, 
        statistic="sum", 
        id_field='location_id'
    )

   # 4. Format time to YYYY-MM
    formatted_time = ds_pixel.time.dt.strftime('%Y-%m').values
    
    # Update coordinates first
    expo_agg_ds = expo_agg_ds.assign_coords({
        "location_id": gdf['location_id'].values,
        "time": formatted_time
    })

    # 5. Robust Renaming
    # If the variable is named 'sum', rename it. 
    # If it's already named 'pop_exposure', we don't need to do anything.
    if "sum" in expo_agg_ds.data_vars:
        expo_agg_ds = expo_agg_ds.rename({"sum": "pop_exposure"})
    elif "pop_exposure" not in expo_agg_ds.data_vars:
        # Fallback: if there is only one data variable, rename it to pop_exposure
        current_vars = list(expo_agg_ds.data_vars)
        if len(current_vars) == 1:
            expo_agg_ds = expo_agg_ds.rename({current_vars[0]: "pop_exposure"})

    return expo_agg_ds


def prepare_boundaries(country: str, level: int) -> GeoDataFrame:
    """Loads, cleans, and standardizes administrative boundaries."""
    logger.info(f'Loading boundaries for {country}-ADM{level}')
    gdf = cgis.io.boundaries.load(country, level=level)

    # Clean columns and standardize ID
    cols_to_drop = ['groups'] 
    gdf = gdf.drop(columns=[c for c in cols_to_drop if c in gdf.columns])
    
    if 'shapeName' in gdf.columns:
        gdf['location_id'] = gdf['shapeName'].astype(str)
    elif 'shapeID' in gdf.columns:
        gdf['location_id'] = gdf['shapeID'].astype(str)
    else:
        gdf['location_id'] = gdf.index.astype(str)

    gdf = gdf[['geometry', 'location_id']]

    # Repair geometries and remove empties
    if not gdf.geometry.is_valid.all():
        logger.warning("Invalid geometries detected. Repairing...")
        gdf.geometry = gdf.geometry.buffer(0)
    
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    return gdf.to_crs("EPSG:4326")


def get_health_data(input_csv: str, regions_df: GeoDataFrame) -> xr.Dataset:
    """Loads health data from CSV or simulates it if missing."""
    if input_csv and Path(input_csv).exists():
        logger.info(f'Loading disease data from {input_csv}')
        df = pd.read_csv(input_csv)
        
        # Standardize columns
        time_col = next((c for c in ["time_period", "time"] if c in df.columns), "time")
        loc_col = next((c for c in ["location", "location_id"] if c in df.columns), "location_id")
        disease_col = next((c for c in ["disease_cases", "disease"] if c in df.columns), "disease")
        
        df = df.rename(columns={time_col: "time", loc_col: "location_id", disease_col: "disease"})
        df["time"] = pd.to_datetime(df["time"])
    else:
        logger.info('Using simulated disease data...')
        df = _simulate_monthly_disease_data(regions_df, 'location_id')

    df['location_id'] = df['location_id'].astype(str)
    return df.set_index(['location_id', 'time']).to_xarray()


def get_environmental_data(country: str, health_xr: xr.Dataset, gdf: GeoDataFrame):
    """Handles loading and aggregation of Population and Temperature."""
    years = np.unique(health_xr.time.dt.year.values).tolist()
    
    # 1. Population
    logger.info(f"Processing population for years: {years}")
    pop_xr = cgis.io.worldpop.load(country_code=country, start=min(years), end=max(years))
    pop_xr.rio.write_crs("EPSG:4326", inplace=True)
    
    # Interpolate to match health data monthly steps
    pop_xr = pop_xr.interp(time=health_xr.time.values, method="linear", kwargs={"fill_value": "extrapolate"})
    pop_agg = aggregate_population_by_year(pop_xr.compute(), gdf)

    # 2. Temperature
    logger.info("Processing monthly temperature...")
    tas_monthly = cgis.io.chelsa.load(
        gdf, start=f"{min(years)}-01", end=f"{max(years)}-12", country_code=country
    )
    tas_monthly.rio.write_crs("EPSG:4326", inplace=True)
    tas_agg = aggregate_temperature_by_month(tas_monthly, gdf)

    return pop_xr, pop_agg, tas_monthly, tas_agg


def dynamic_periods(
    country: str,
    level: int = 5,
    input_csv: str = 'data/inputs/disease-data.csv',
    out_path: Path = Path("data/outputs/dynamic_health.csv"),
) -> None:
    """Orchestrates the full data pipeline."""
    # 1. Setup Base Data
    gdf = prepare_boundaries(country, level)
    health_xr = get_health_data(input_csv, gdf)

    # 2. Get Environmental Rasters and Aggregates
    pop_native, pop_agg, tas_native, tas_agg = get_environmental_data(country, health_xr, gdf)

    # 3. Calculate Exposure
    exposure_ds = _calculate_monthly_exposure_from_vars(
        aoi=gdf, gdf=gdf, pop_native=pop_native, 
        tas_native=tas_native, country_code=country,
        params=MalariaExposureParams(resolution_m=100.0)
    )

    # 4. Alignment and Merging
    logger.info("Merging all datasets...")
    # Standardize time strings for a clean merge
    time_coords = health_xr.time.dt.strftime('%Y-%m').values
    health_xr = health_xr.assign_coords(time=time_coords)
    pop_agg = pop_agg.assign_coords(time=time_coords)

    final_ds = xr.merge([health_xr, pop_agg, tas_agg, exposure_ds])
    final_df = final_ds.to_dataframe().reset_index()

    # 5. Add Centroids and Save
    centroids = gdf.copy()
    centroids['geometry'] = centroids.geometry.representative_point()
    centroids['latitude'], centroids['longitude'] = centroids.geometry.y, centroids.geometry.x
    
    final_df = final_df.merge(
        centroids[['location_id', 'latitude', 'longitude']], 
        on='location_id', how='left'
    )

    final_df.to_csv(out_path, index=False)
    logger.info(f"Pipeline complete. Dataset saved to {out_path}")