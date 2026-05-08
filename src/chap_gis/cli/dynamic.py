from pathlib import Path
from geopandas import GeoDataFrame
import pandas as pd
import pandas
from typing import Optional
import xarray as xr
import rioxarray
import numpy as np
import chap_gis as cgis

import logging

logger = logging.getLogger(__name__)
from chap_gis.aggregate import aggregate_to_regions
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

def _aggregate_population_by_year(pop_da: xr.DataArray, gdf: GeoDataFrame) -> xr.Dataset:
    """
    Aggregates population raster data to administrative regions.
    
    This function handles:
    1. CRS verification for spatial alignment.
    2. Zonal statistics (sum) across all time steps.
    3. Restoring location names from the GeoDataFrame.
    4. Restoring actual year/time labels from the input raster.
    5. Standardizing variable names for downstream merging.
    """
    logger.info("Starting regional population aggregation...")
    
    # 1. Coordinate Reference System (CRS) check
    # exact_extract requires the raster and gdf to have defined CRSs.
    if pop_da.rio.crs is None:
        logger.warning("Raster CRS missing. Defaulting to EPSG:4326.")
        pop_da = pop_da.rio.write_crs("EPSG:4326")
    
    # 2. Execute Zonal Statistics
    # aggregate_to_regions returns dims ['location_id', 'time'] 
    # but initially uses integer indices (0, 1, 2...) for coordinates.
    agg_ds = aggregate_to_regions(
        pop_da, 
        gdf, 
        statistic="sum", 
        id_field='location_id'
    )

    # 3. Restore Coordinate Labels
    # We replace the integer indices with the actual metadata from our inputs.
    agg_ds = agg_ds.assign_coords({
        "location_id": gdf['location_id'].values,
        "time": pop_da.time.values
    })

    # 4. Standardize Variable Names
    # exact_extract outputs the name of the statistic ('sum'). 
    # We rename it to 'population' to match health data expectations.
    if "sum" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"sum": "population"})

    # 5. Clean up Metadata
    # Remove any residual 'band' coordinates if they exist
    if 'band' in agg_ds.coords:
        agg_ds = agg_ds.drop_vars('band')

    logger.info(f"Aggregation complete for {len(agg_ds.location_id)} regions.")
    return agg_ds

def _aggregate_temperature_by_month(tas_da: xr.DataArray, gdf: GeoDataFrame) -> xr.Dataset:
    """
    Aggregates CHELSA monthly temperature rasters to administrative regions.
    
    This function handles:
    1. CRS verification.
    2. Zonal statistics (mean) across all monthly time steps.
    3. Restoring location and time metadata (formatted as YYYY-MM).
    4. Renaming the output variable to 'tas'.
    """
    logger.info("Starting regional temperature aggregation...")
    
    # 1. Coordinate Reference System (CRS) check
    if tas_da.rio.crs is None:
        logger.warning("Temperature raster CRS missing. Defaulting to EPSG:4326.")
        tas_da = tas_da.rio.write_crs("EPSG:4326")
    
    # 2. Execute Zonal Statistics
    agg_ds = aggregate_to_regions(
        tas_da, 
        gdf, 
        statistic="mean", 
        id_field='location_id'
    )

    # 3. Restore and Format Coordinate Labels
    # We convert the datetime objects to 'YYYY-MM' strings
    formatted_time = tas_da.time.dt.strftime('%Y-%m').values
    
    agg_ds = agg_ds.assign_coords({
        "location_id": gdf['location_id'].values,
        "time": formatted_time
    })

    # 4. Standardize Variable Names
    if "mean" in agg_ds.data_vars:
        agg_ds = agg_ds.rename({"mean": "tas"})

    # 5. Clean up Metadata
    if 'band' in agg_ds.coords:
        agg_ds = agg_ds.drop_vars('band')

    logger.info(f"Aggregation complete for {len(agg_ds.location_id)} regions.")
    return agg_ds

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


def dynamic_periods (
    country: str,
    level: int = 5,
    input_csv: str = 'data/inputs/chap_data_levl5.csv',
    out_path: Path = Path("data/outputs/augmented-pop-exposure_dynamic_health.csv"),
)-> None:
    """
    Placeholder for dynamic data processing function.
    """
    logger.info(f"Running dynamic data processing for {country} at level {level}")
    logger.info("This function is a placeholder and should be implemented with actual logic.")

    logger.info(f'Loading boundaries for {country}-ADM{level}')
    gdf = cgis.io.boundaries.load(country, level=level)

    # exactextract cannot handle array datatypes in element groups
    cols_to_drop = ['groups'] 
    gdf = gdf.drop(columns=[c for c in cols_to_drop if c in gdf.columns])
    
    # Standardize boundary ID to 'location_id' (forced to string for alignment)
    if 'shapeName' in gdf.columns:
        gdf['location_id'] = gdf['shapeName'].astype(str)
    elif 'shapeID' in gdf.columns:
        gdf['location_id'] = gdf['shapeID'].astype(str)
    else:
        gdf['location_id'] = gdf.index.astype(str)

    gdf = gdf[['geometry', 'location_id']]

    # Fix self-intersections or invalid rings
    if not gdf.geometry.is_valid.all():
        logger.warning("Invalid geometries detected. Attempting to repair...")
        gdf.geometry = gdf.geometry.buffer(0)

    # REMOVE EMPTY: Drop rows with missing geometry
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

    # 2. Load/Generate Health Data
    if input_csv and Path(input_csv).exists():
        logger.info(f'Loading disease data from {input_csv}')
        health_df = pd.read_csv(input_csv)
        
        # Standardize columns for merging logic
        time_col = "time_period" if "time_period" in health_df.columns else "time"
        loc_col = "location" if "location" in health_df.columns else "location_id"
        disease_col = "disease_cases" if "disease_cases" in health_df.columns else "disease"
        
        health_df = health_df.rename(columns={
            time_col: "time",
            loc_col: "location_id",
            disease_col: "disease"
        })
        health_df["time"] = pd.to_datetime(health_df["time"])
    else:
        logger.info('Using simulated disease data...')
        health_df = _simulate_monthly_disease_data(gdf, 'location_id')

    # Force string type on join key to avoid DType conflicts
    health_df['location_id'] = health_df['location_id'].astype(str)
    health_xr = health_df.set_index(['location_id', 'time']).to_xarray()

    # Log unique years in the health data 
    uniq_years = np.unique(health_xr.time.dt.year.values)
    year_list = uniq_years.tolist()
    logger.info(f"Unique years in health data: {year_list}")

    logger.info("Merging health data with population data...")                                   
    pop_xr = cgis.io.worldpop.load(country_code=country,start=min(year_list),end=max(year_list))

    logger.info("loading population data complete. Here are the details:")
    
    #interpolate the populateion data to monthly frequency to match the health data time steps
    pop_xr = pop_xr.interp(time=health_xr.time.values, method="linear", kwargs={"fill_value": "extrapolate"})
    logger.info(pop_xr)

    # Aggregate population to regions
    pop_xr_loaded = pop_xr.compute()
    pop_agg_ds = _aggregate_population_by_year(pop_xr_loaded, gdf)
    logger.info("Population aggregation complete. ")
    
    # Temperature data loading and aggregation would go here
    tasMonthly = cgis.io.chelsa.load(
        gdf,
        start=f"{min(year_list)}-01",
        end=f"{max(year_list)}-12",
        country_code=country,
    )

    logger.info("Temperature data loading complete. Here are the details:")
    logger.info(tasMonthly)

    tas_agg_ds = _aggregate_temperature_by_month(tasMonthly, gdf)
    logger.info("Temperature aggregation complete.")

    # Calculate exposure using the already loaded and aggregated population and temperature data
    exposure_ds = _calculate_monthly_exposure_from_vars(
        aoi=gdf,
        gdf=gdf,
        pop_native=pop_xr,
        tas_native=tasMonthly,
        country_code=country,
        params=MalariaExposureParams(resolution_m=30.0)
    )

    exposure_df = exposure_ds.to_dataframe().reset_index()

    # 2. Define the output path
    exposure_csv_path = out_path

    # 3. Save to CSV
    exposure_df.to_csv(exposure_csv_path, index=False)

    logger.info(f"Monthly exposure data saved to {exposure_csv_path}")