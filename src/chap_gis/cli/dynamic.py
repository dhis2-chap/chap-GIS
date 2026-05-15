from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import chap_gis as cgis
from tqdm import tqdm

from chap_gis.aggregate import (
    aggregate_population_by_year,
    aggregate_temperature_by_month,
    aggregate_to_regions,
)

from chap_gis.pipelines.malaria_exposure import (
    run as run_exposure_pipeline,
    MalariaExposureParams,
)

logger = logging.getLogger(__name__)

# Balanced chunk size for 30m resolution data
CHUNKS = {"time": 1, "x": 1024, "y": 1024}

def chunk(ds):
    """Apply safe chunking only on valid dims and ensure Dask-backing."""
    if ds is None:
        return None
    valid = {k: v for k, v in CHUNKS.items() if k in ds.dims}
    return ds.chunk(valid) if valid else ds

def normalize(ds: xr.Dataset | xr.DataArray, name: str) -> xr.Dataset:
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=name)

    if "time" in ds.coords:
        new_time = pd.to_datetime(ds.time.values)
        if new_time.year.min() < 1990:
            logger.warning(f"Detected suspicious '1970' timestamps in {name}.")
        ds = ds.assign_coords(time=new_time.astype("datetime64[ns]"))

    return ds

def prepare_boundaries(country: str, level: int) -> gpd.GeoDataFrame:
    gdf = cgis.io.boundaries.load(country, level=level)
    gdf["location_id"] = gdf.get("shapeName", gdf.index).astype(str)
    gdf = gdf[["geometry", "location_id"]].to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    if not gdf.geometry.is_valid.all():
        logger.warning("Invalid geometries detected. Applying buffer(0) fix.")
        gdf["geometry"] = gdf.geometry.buffer(0)

    return gdf

def _simulate_monthly_disease_data(regions_df: pd.DataFrame, location_id_field: str) -> pd.DataFrame:
    df = regions_df[[location_id_field]].copy()
    df = df.rename(columns={location_id_field: 'location_id'}).drop_duplicates()
    time_df = pd.DataFrame({"time": pd.date_range("2017-01-01", periods=36, freq="MS")})
    final = df.merge(time_df, how='cross')
    final['disease'] = np.random.uniform(0, 30, size=len(final))
    return final

def get_health_data(input_csv: str, gdf: gpd.GeoDataFrame) -> xr.Dataset:
    if input_csv and Path(input_csv).exists():
        df = pd.read_csv(input_csv)
        mapping = {
            next(c for c in ["time", "date", "time_period"] if c in df): "time",
            next(c for c in ["location_id", "location"] if c in df): "location_id",
            next(c for c in ["disease", "disease_cases"] if c in df): "disease",
        }
        df = df.rename(columns=mapping)
        df["time"] = pd.to_datetime(df["time"])
    else:
        df = _simulate_monthly_disease_data(gdf, "location_id")

    df = df.dropna(subset=["location_id", "time", "disease"])
    ds = df.set_index(["location_id", "time"]).to_xarray()
    return normalize(ds, "disease")

def get_environmental_data(country, health, gdf, inter):
    years = sorted(health.time.dt.year.to_series().unique().tolist())
    
    # 1. Population: Load and immediately chunk to keep it lazy
    pop_xr = cgis.io.worldpop.load(country_code=country, start=min(years), end=max(years))
    pop_xr = chunk(pop_xr)
    pop_xr.rio.write_crs("EPSG:4326", inplace=True)
    
    if inter:
        pop_xr = pop_xr.interp(time=health.time, method="linear", kwargs={"fill_value": "extrapolate"})
    else:
        pop_xr = pop_xr.reindex(time=health.time, method="ffill").bfill(dim="time")

    # 2. Aggregation: Only compute if the aggregation function requires it. 
    # If aggregate_population_by_year is not Dask-aware, we compute here.
    logger.info("Aggregating population...")
    pop_agg = aggregate_population_by_year(pop_xr, gdf)
    
    if "population" not in pop_agg.data_vars:
        var_name = list(pop_agg.data_vars)[0]
        pop_agg = pop_agg.rename({var_name: "population"})

    # 3. Temperature: Load and chunk
    tas = cgis.io.chelsa.load(gdf, start=f"{min(years)}-01", end=f"{max(years)}-12", country_code=country)
    tas = chunk(tas)
    
    for dim in ["x", "y"]:
        if dim in tas.coords:
            tas[dim] = np.round(tas[dim].astype("float64"), 10)
            
    tas.rio.write_crs("EPSG:4326", inplace=True)

    logger.info("Aggregating temperature...")
    # aggregate_temperature_by_month usually works better on computed/small data
    tas_agg = normalize(aggregate_temperature_by_month(tas, gdf), "tas")

    return pop_xr, pop_agg, tas, tas_agg

def _run_core_logic(gdf, health, pop, tas, country, params):
    years = np.unique(health.time.dt.year.values)
    results = []

    worldcover_year = max(2020, min(int(health.time.dt.year.max()), 2021))
    aoi = cgis.aoi.buffered(gdf, params.aoi_buffer_deg)
    
    # Chunk static layers
    land = chunk(cgis.io.worldcover.load(aoi=aoi, start=worldcover_year, end=worldcover_year, country_code=country))
    elev = chunk(cgis.io.elevation.load(aoi=aoi, country_code=country))
    rice = chunk(cgis.io.rice.load(country_code=country))

    for y in tqdm(years, desc=f"Processing Years for {country}"):
        logger.info(f"Processing year {y} for {country}...")
        # Slicing keeps the Dask chunks intact
        tas_y = tas.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
        pop_y = pop.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))

        if tas_y.time.size == 0:
            continue

        # Pipeline runs lazily if inputs are Dask-backed
        ds = run_exposure_pipeline(
            aoi=gdf, 
            landcover_native=land, 
            elev_native=elev,
            tas_monthly=tas_y, 
            population_native=pop_y, 
            rice_native=rice, 
            params=params,
        )
        
        expo_raster = ds["pop_exposure"]
        
        # Aggregate to regions (this is often the 'compute' trigger)
        expo_out = aggregate_to_regions(
            expo_raster,
            gdf,
            statistic="sum",
            id_field="location_id",
        )

        if isinstance(expo_out, xr.DataArray):
            expo_y = expo_out.to_dataset(name="pop_exposure")
        else:
            expo_y = expo_out.rename({"values": "pop_exposure"}) if "values" in expo_out.data_vars else expo_out

        expo_y = expo_y.assign_coords(time=tas_y.time)
        results.append(expo_y)

    return xr.concat(results, dim="time") if results else None

def dynamic_periods(
    country: str,
    level: int = 5,
    inter: bool = True,
    input_csv: str = "./data/inputs/disease-data.csv",
    out_path: Path = Path("test.csv"),
):
    logger.info(f"Starting pipeline for {country}")

    gdf = prepare_boundaries(country, level)
    health = get_health_data(input_csv, gdf)
    pop, pop_agg, tas, tas_agg = get_environmental_data(country, health, gdf, inter)

    params = MalariaExposureParams(resolution_m=30.0)
    expo = _run_core_logic(gdf, health, pop, tas, country, params)
    
    logger.info("Merging regional datasets...")
    # These are already aggregated to regions (small), so merge is safe
    datasets_to_merge = [ds for ds in [health, tas_agg, pop_agg, expo] if ds is not None]
    final = xr.merge(datasets_to_merge, join="inner")

    # Final export to pandas (Safe only because these are region-level aggregates)
    df = final.to_dataframe().reset_index()

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m")
        cols_to_keep = ["location_id", "time", "disease", "tas", "population", "pop_exposure"]
        df = df[[c for c in cols_to_keep if c in df.columns]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Done. Rows written: {len(df)}")