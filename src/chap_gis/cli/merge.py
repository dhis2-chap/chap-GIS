import logging
from pathlib import Path

import pandas as pd
import xarray as xr
import rioxarray
import numpy as np
import chap_gis as cgis
from chap_gis.aggregate import aggregate_to_regions
from dhis2eo.integrations.chap import dataframe_to_chap_csv

logger = logging.getLogger(__name__)

def _wrapper(raster_dir: str, gdf: pd.DataFrame) -> xr.Dataset:
    """
    Aggregates all NetCDFs into a single xarray Dataset indexed by location_id.
    Removes all temporal information to ensure it can broadcast across health data.
    """
    raster_path = Path(raster_dir)
    ds_list = []

    stats_map = {
        "population": "sum",
        "pop_exposure": "sum",
        "breeding": "mean",     
        "suitability": "mean",
        "expo": "mean",
        "temperature": "mean",
        "elevation": "mean"
    }

    for path in raster_path.glob('*.nc'):
        var_name = path.stem
        stat_to_use = stats_map.get(var_name, "mean")
        
        logger.info(f'Processing {var_name} using statistic: {stat_to_use}')
        
        raster = rioxarray.open_rasterio(path)
        
        raster = raster.squeeze(drop=True)
        if 'time' in raster.dims:
            raster = raster.mean(dim='time')
        
        if 'time' in raster.coords:
            raster = raster.drop_vars('time')

        agg_obj = aggregate_to_regions(
            raster, 
            gdf, 
            statistic=stat_to_use, 
            id_field='location_id'
        )
        
        if isinstance(agg_obj, xr.DataArray):
            agg_obj.name = var_name
            agg_ds = agg_obj.to_dataset()
        else:
            mapping = {var: var_name for var in agg_obj.data_vars}
            agg_ds = agg_obj.rename(mapping)
            
        ds_list.append(agg_ds)

    if not ds_list:
        raise FileNotFoundError(f"No .nc files found in {raster_dir}")

    return xr.merge(ds_list)

def _simulate_monthly_disease_data(regions_df: pd.DataFrame, location_id_field: str) -> pd.DataFrame:
    """
    Generates dummy disease data for the given regions.
    """
    df = regions_df[[location_id_field]].copy()
    df = df.rename(columns={location_id_field: 'location_id'})
    df = df.drop_duplicates(subset=["location_id"])

    # Create 3 years of monthly timestamps
    time_df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=36, freq="MS")})

    # Cross join to create (location x time) grid
    final = df.merge(time_df, how='cross')
    final['disease'] = np.random.uniform(0, 30, size=len(final))

    return final

def merge(
    raster_dir: str,
    country: str,
    level: int,
    input_csv: str = None, 
    out_path: Path = Path("data/outputs/augmented_health.csv"),
) -> None:
    """
    Converts health CSV to xarray, merges with static environmental data, 
    and exports the result back to a flattened CHAP-compatible CSV.
    """
    # 1. Load Admin Boundaries
    logger.info(f'Loading boundaries for {country}-ADM{level}')
    gdf = cgis.io.boundaries.load(country, level=level)
    
    # Standardize boundary ID to 'location_id' (forced to string for alignment)
    if 'shapeName' in gdf.columns:
        gdf['location_id'] = gdf['shapeName'].astype(str)
    elif 'shapeID' in gdf.columns:
        gdf['location_id'] = gdf['shapeID'].astype(str)
    else:
        gdf['location_id'] = gdf.index.astype(str)

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

    # 3. Aggregate Environmental Data
    env_ds = _wrapper(raster_dir, gdf)

    # Population-weighted mean exposure index per person in each region:
    # Σ(pop·expo) / Σ(pop); guard regions with zero population.
    env_ds["mean_exposure_per_person"] = (
        env_ds["pop_exposure"] / env_ds["population"].where(env_ds["population"] > 0)
    )

    # Ensure env_ds uses string coordinate and has NO time dimension
    if 'location_id' in env_ds.coords:
        env_ds.coords['location_id'] = env_ds.coords['location_id'].astype(str)
    
    # Final safety check to remove accidental time coordinates from static data
    if 'time' in env_ds.coords or 'time' in env_ds.dims:
        env_ds = env_ds.drop_vars('time', errors='ignore').squeeze(drop=True)

    # 4. Merge
    logger.info("Merging datasets...")
    # join='left' preserves the time index from health_xr and broadcasts env_ds across it
    final_ds = xr.merge([health_xr, env_ds], join='left')

    # 5. Flatten and Export
    final_df = final_ds.to_dataframe().reset_index()
    
    column_map = {
        "time_period": "time",
        "location": "location_id",
        "disease_cases": "disease",
        "population": "population",
    }

    if "population" not in final_df.columns:
        column_map.pop("population")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing validated CHAP CSV to {out_path}")

    dataframe_to_chap_csv(
        df=final_df,
        column_map=column_map,
        freq="monthly",
        output_path=str(out_path),
    )
    
    logger.info("Process successful.")