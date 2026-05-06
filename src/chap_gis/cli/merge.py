import xarray as xr
import rioxarray
import pandas as pd
import logging
from pathlib import Path
import chap_gis as cgis
from chap_gis.aggregate import aggregate_to_regions
import numpy as np

logger = logging.getLogger(__name__)

def wrapper(raster_dir: str, gdf: pd.DataFrame) -> xr.Dataset:
    """
    Aggregates all NetCDFs into a single xarray Dataset indexed by location_id.
    Applies specific zonal statistics (sum vs mean) based on variable type.
    """
    raster_path = Path(raster_dir)
    ds_list = []

    # Map specific layers to 'sum', default everything else to 'mean'
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
        stat_to_use = stats_map.get(var_name, "mean") # Fallback to mean
        
        logger.info(f'Processing {var_name} using statistic: {stat_to_use}')
        
        raster = rioxarray.open_rasterio(path)
        if 'time' in raster.dims:
            raster = raster.mean(dim='time')
        raster = raster.squeeze(drop=True)

        # Aggregate pixels using the specific statistic
        agg_obj = aggregate_to_regions(
            raster, 
            gdf, 
            statistic=stat_to_use, 
            id_field='location_id'
        )
        
        # Deduplicate location_id if necessary
        # Note: If stat was sum, we should sum duplicates; if mean, mean them.
        if 'location_id' in agg_obj.coords:
            agg_obj = (
                agg_obj.groupby('location_id').sum() 
                if stat_to_use == "sum" 
                else agg_obj.groupby('location_id').mean()
            )

        # Rename to clean header
        if isinstance(agg_obj, xr.Dataset):
            mapping = {var: var_name for var in agg_obj.data_vars}
            agg_ds = agg_obj.rename(mapping)
        else:
            agg_obj.name = var_name
            agg_ds = agg_obj.to_dataset()
            
        ds_list.append(agg_ds)

    return xr.merge(ds_list)

def _simulate_monthly_disease_data(regions_df, location_id_field):
    # set location id field
    regions_df['location_id'] = regions_df[location_id_field]
    regions_df = regions_df[['location_id']]
    regions_df = regions_df.drop_duplicates(subset=["location_id"])

    #set population to random number for now
    pop_df = regions_df.copy()
    pop_df["population"] = np.random.randint(1000, 500000, size=len(pop_df))

    # create df for timeframe (hardcoded for now)
    time_df = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=12*3, freq="MS")})

    # crossjoin regions with time
    final = pop_df.merge(time_df, how='cross')

    # add random disease data
    from random import uniform
    final['disease'] = [uniform(0, 30) for _ in range(len(final))]

    return final

def merge(
    input_csv: str,
    raster_dir: str,
    country: str,
    level: int,
    out_path: Path = Path("data/outputs/augmented_health.csv"),
) -> None:
    """
    Converts health CSV to xarray, merges with static environmental data, 
    and exports the result back to a flattened CSV.
    """
    # 1. Load health data and pivot to xarray
    health_df = pd.read_csv(input_csv)
    health_xr = health_df.set_index(['location', 'time_period']).to_xarray()
    
    # 2. Load geographic boundaries
    gdf = cgis.io.boundaries.load(country, level=level)
    
    # Map boundary names to 'location_id' for coordinate alignment
    if 'shapeName' in gdf.columns:
        gdf['location_id'] = gdf['shapeName']
    elif 'shapeID' in gdf.columns:
        gdf['location_id'] = gdf['shapeID']
    else:
        gdf['location_id'] = gdf.index.astype(str)

    # 3. Get aggregated environmental data
    env_ds = wrapper(raster_dir, gdf)

    # 4. Align coordinates and Merge
    # Rename 'location_id' to match 'location' dimension in health data
    env_ds = env_ds.rename({'location_id': 'location'})
    
    # xarray.merge handles the broadcasting of static data across time automatically
    final_ds = xr.merge([health_xr, env_ds])

    # 5. Flatten and Export
    final_df = final_ds.to_dataframe().reset_index()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info(f"Augmented CSV created at {out_path}")