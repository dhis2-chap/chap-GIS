"""``dynamic_periods`` subcommand: multi-month, multi-year exposure pipeline.

TODO(testability) — refactor this module so the orchestration is testable
without monkeypatching six different ``chap_gis.io.*`` loaders:

    1. Extract a pure inner function::

           def _run(
               *,
               gdf: GeoDataFrame,
               health_xr: xr.Dataset,
               pop_native: xr.DataArray,
               tas_native: xr.DataArray,
               landcover_native: xr.DataArray,
               elev_native: xr.DataArray,
               rice_native: xr.DataArray,
               params: MalariaExposureParams,
           ) -> xr.Dataset: ...

       ``dynamic_periods`` becomes thin: load → _run → write CSV. The inner
       function is drivable with synthetic DataArrays — no patching needed.
       This mirrors how ``pipelines/malaria_exposure.run`` is already split
       out from ``cli/analyze.py``.
    2. Hoist the static-layer loads (worldcover, elevation, rice) out of the
       per-year loop in ``_calculate_monthly_exposure_from_vars`` — they do
       not change between years.
    3. Promote ``MalariaExposureParams`` (currently constructed inline with
       ``resolution_m=30.0``) to a CLI option so tests can pass a coarser
       resolution and the full pipeline can run on tiny synthetic AOIs in
       under a second.

KNOWN CORRECTNESS CONCERNS — each is targeted by a test in
``tests/test_cli_dynamic.py`` that is expected to fail today and turn green
when the bug is fixed:

    1. **Time dtype mismatch silently empties the final inner merge.**
       ``aggregate_temperature_by_month`` (in ``chap_gis.aggregate``) and
       ``_calculate_monthly_exposure_from_vars`` (below) cast ``time`` to
       ``YYYY-MM`` strings via ``dt.strftime``. ``health_xr`` and ``pop_agg``
       keep ``datetime64``. The closing
       ``xr.merge([..., join="inner"])`` therefore aligns nothing on the
       time axis and the output CSV is empty or all-NaN.
       Reveal: assert ``np.issubdtype(agg.time.dtype, np.datetime64)`` on
       the aggregators, and assert the end-to-end CSV has
       ``len(df) == n_locations * n_months`` rows.

    2. **Static layers reloaded every year.**
       worldcover, elevation, and rice are loaded once *per year* inside the
       loop. They are static across years (and worldcover is in fact clamped
       to 2020/2021 — see (3)).
       Reveal: monkeypatch each loader with a call counter and assert it is
       called exactly once across a multi-year run.

    3. **WorldCover year silently clamped to 2020–2021.**
       ``worldcover_year = max(2020, min(requested_year, 2021))`` returns
       2020 for any year ≤ 2020 and 2021 for any year ≥ 2021 with only an
       INFO log. Outputs are then mis-labelled with respect to the year the
       caller asked for.
       Reveal: parametrize the inner function with
       ``requested_year ∈ {2018, 2020, 2021, 2025}`` and assert either an
       explicit warning, or (after refactor) that the resolved year is a
       caller-supplied parameter.

    4. **``pop_xr.compute()`` materializes the full raster early.**
       For multi-year, country-scale rasters this is a memory cliff that
       defeats the laziness ``run_exposure_pipeline`` carefully preserves.
       The ``gc.collect()`` calls in the loop are the symptom.
       Reveal: harder to unit-test; an integration test on a moderately
       sized country watched for peak RSS would catch it.

    5. **``aoi=gdf`` vs ``aoi=country_geom``.**
       ``dynamic_periods`` passes the level-N admin GDF as ``aoi`` into the
       exposure pipeline, which uses it to build the grid. ``cli/analyze``
       uses level-0 boundaries here. The difference is inefficient at best
       (the grid is built from the union of every admin polygon) and may
       silently change the grid extent at worst.
       Reveal: assert the grid extent built inside the pipeline matches the
       expected country bbox rather than the admin GDF's geometry union.
"""

import logging
from pathlib import Path
from typing import Optional

import chap_gis as cgis
import numpy as np
import pandas as pd
import xarray as xr
from geopandas import GeoDataFrame
from chap_gis.aggregate import aggregate_population_by_year, aggregate_temperature_by_month, aggregate_to_regions
from chap_gis.pipelines.malaria_exposure import run as run_exposure_pipeline
from chap_gis.pipelines.malaria_exposure import MalariaExposureParams

logger = logging.getLogger(__name__)

def _simulate_monthly_disease_data(regions_df: pd.DataFrame, location_id_field: str) -> pd.DataFrame:
    """Generates dummy disease data with strict datetime64 timestamps."""
    df = regions_df[[location_id_field]].copy()
    df = df.rename(columns={location_id_field: 'location_id'}).drop_duplicates()
    
    # Use freq="MS" for Month Start to ensure clean alignment
    time_df = pd.DataFrame({"time": pd.date_range("2017-01-01", periods=36, freq="MS")})
    final = df.merge(time_df, how='cross')
    final['disease'] = np.random.uniform(0, 30, size=len(final))
    return final

def prepare_boundaries(country: str, level: int) -> GeoDataFrame:
    """Loads and standardizes administrative boundaries."""
    gdf = cgis.io.boundaries.load(country, level=level)
    
    if 'shapeName' in gdf.columns:
        gdf['location_id'] = gdf['shapeName'].astype(str)
    else:
        gdf['location_id'] = gdf.index.astype(str)

    gdf = gdf[['geometry', 'location_id']].to_crs("EPSG:4326")
    if not gdf.geometry.is_valid.all():
        gdf.geometry = gdf.geometry.buffer(0)
    
    return gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

def get_health_data(input_csv: str, regions_df: GeoDataFrame) -> xr.Dataset:
    """Loads health data and ensures variable naming and datetime types."""
    if input_csv and Path(input_csv).exists():
        df = pd.read_csv(input_csv)
        
        # Mapping to satisfy: test_get_health_data_csv_renames_chap_columns
        time_col = next((c for c in ["time_period", "time", "date"] if c in df.columns), "time")
        loc_col = next((c for c in ["location_id", "location"] if c in df.columns), "location_id")
        disease_col = next((c for c in ["disease_cases", "disease"] if c in df.columns), "disease")
        
        df = df.rename(columns={time_col: "time", loc_col: "location_id", disease_col: "disease"})
        # Force conversion to datetime64
        df["time"] = pd.to_datetime(df["time"])
    else:
        df = _simulate_monthly_disease_data(regions_df, 'location_id')

    df['location_id'] = df['location_id'].astype(str)
    
    # Convert to dataset; if it's a series, it becomes a DataArray, so we name it 'disease'
    ds = df.set_index(['location_id', 'time']).to_xarray()
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name="disease")
    elif "disease" not in ds.data_vars and len(ds.data_vars) == 1:
        # If the CSV had a weird name like 'cases', rename it to 'disease'
        ds = ds.rename({list(ds.data_vars)[0]: "disease"})
    return ds

def get_environmental_data(country: str, health_xr: xr.Dataset, gdf: GeoDataFrame, inter: bool):
    """Fetches environmental data, ensuring coordinate type consistency."""
    years = np.unique(health_xr.time.dt.year.values).tolist()
    
    # Population
    pop_xr = cgis.io.worldpop.load(country_code=country, start=min(years), end=max(years))
    pop_xr.rio.write_crs("EPSG:4326", inplace=True)
    
    # Use health_xr.time directly to ensure exact type match (datetime64)
    if inter:
        pop_xr = pop_xr.interp(time=health_xr.time, method="linear", kwargs={"fill_value": "extrapolate"})
    else:
        pop_xr = pop_xr.reindex(time=health_xr.time, method="ffill").bfill(dim="time")

    pop_agg = aggregate_population_by_year(pop_xr.compute(), gdf)
    # Standardize population variable name
    if "population" not in pop_agg.data_vars:
        pop_agg = pop_agg.rename({list(pop_agg.data_vars)[0]: "population"})

    # Temperature
    tas_monthly = cgis.io.chelsa.load(gdf, start=f"{min(years)}-01", end=f"{max(years)}-12", country_code=country)
    tas_monthly.rio.write_crs("EPSG:4326", inplace=True)
    tas_monthly = tas_monthly.assign_coords(
        time=pd.to_datetime(tas_monthly.time.values)
    )

    tas_agg = aggregate_temperature_by_month(tas_monthly, gdf)

    tas_agg = tas_agg.assign_coords(
        time=pd.to_datetime(tas_agg.time.values)
    )
    
    # Standardize to 'tas' to satisfy test_dynamic_periods_writes_full_csv
    if "tas" not in tas_agg.data_vars:
        tas_agg = tas_agg.rename({list(tas_agg.data_vars)[0]: "tas"})

    return pop_xr, pop_agg, tas_monthly, tas_agg

def _run_core_logic(gdf, health_xr, pop_native, tas_native, landcover_native, elev_native, rice_native, params) -> xr.Dataset:
    """Core iteration logic preserving datetime64 indices."""
    years = sorted(np.unique(health_xr.time.dt.year.values).tolist())
    yearly_results = []

    for year in years:
        # Selection using strings is fine, it maintains the underlying datetime64 index
        logger.info(f"Processing year {year}")
        pop_year = pop_native.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        tas_year = tas_native.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

        logger.info("Running exposure pipeline for the year")
        ds_pixel = run_exposure_pipeline(
            aoi=gdf, landcover_native=landcover_native, elev_native=elev_native,
            tas_monthly=tas_year, population_native=pop_year, rice_native=rice_native, params=params,
        ).compute()

        logger.info("Aggregating exposure results to regions")
        expo_agg = aggregate_to_regions(ds_pixel["pop_exposure"], gdf, statistic="sum", id_field='location_id')

        if isinstance(expo_agg, xr.DataArray):
            expo_agg = expo_agg.to_dataset(name="pop_exposure")

        if "sum" in expo_agg.data_vars:
            expo_agg = expo_agg.rename({"sum": "pop_exposure"})

        expo_agg = expo_agg.assign_coords({
        "location_id": gdf['location_id'].values,
        "time": pd.to_datetime(ds_pixel.time.values)
        })
        
        # KEY FIX: Explicitly assign the datetime64 time coordinate from the pixel results
        expo_agg = expo_agg.assign_coords({
            "location_id": gdf['location_id'].values,
            "time": ds_pixel.time.values  # This is datetime64[ns]
        })


        if "sum" in expo_agg.data_vars:
            expo_agg = expo_agg.rename({"sum": "pop_exposure"})
            
        yearly_results.append(expo_agg)

    exposure_ds = xr.concat(yearly_results, dim="time")

    #safe gaurd against unnamed variable showing up as 'values'
    if isinstance(exposure_ds, xr.DataArray):
        exposure_ds = exposure_ds.to_dataset(name="pop_exposure")

    if "sum" in exposure_ds.data_vars:
        exposure_ds = exposure_ds.rename({"sum": "pop_exposure"})

    if "values" in exposure_ds.data_vars:
        exposure_ds = exposure_ds.rename({"values": "pop_exposure"})

    return exposure_ds

def dynamic_periods(
    country: str,
    level: int = 5,
    inter: bool = True,
    input_csv: str = "data/inputs/disease-data.csv",
    out_path: Path = Path("data/outputs/health_pipeline_output.csv"),
) -> None:
    logger.info(f"Running dynamic_periods for {country} at level {level} with inter={inter}")
    gdf = prepare_boundaries(country, level)
    health_xr = get_health_data(input_csv, gdf)
    
    logger.info(f"Loading environmental data for years")
    pop_native, pop_agg, tas_native, tas_agg = get_environmental_data(country, health_xr, gdf, inter)

    params = MalariaExposureParams(resolution_m=30.0)
    logger.info("loading static layers (worldcover, elevation, rice)")
    worldcover_year = max(2020, min(int(health_xr.time.dt.year.max()), 2021))
    buffered_aoi = cgis.aoi.buffered(gdf, distance=params.aoi_buffer_deg)
    
    landcover_native = cgis.io.worldcover.load(buffered_aoi, start=worldcover_year, end=worldcover_year, country_code=country)
    elev_native = cgis.io.elevation.load(buffered_aoi, country_code=country)
    rice_native = cgis.io.rice.load(country_code=country)

    logger.info("Running core exposure logic")
    exposure_ds = _run_core_logic(gdf, health_xr, pop_native, tas_native, landcover_native, elev_native, rice_native, params)

    # All components MUST have datetime64 'time' for this merge to succeed
    logger.info("Merging datasets on location_id and time")
    final_ds = xr.merge([health_xr, tas_agg, pop_agg, exposure_ds], join="inner")

    # Convert to dataframe and reset index to turn coordinates into columns
    final_df = final_ds.to_dataframe().reset_index()
    logger.info("Converting merged dataset to DataFrame")
    # Fix 'values' column issue: If any DataArray wasn't named, it shows up as 'values'
    if 'values' in final_df.columns:
        # Attempt to recover name or drop it if it's redundant
        logger.warning("Found 'values' column in output. Check variable naming in pipeline.")

    # FINAL STEP ONLY: Format time to string for the CSV file
    # Doing this earlier causes the 'AssertionError: expected datetime64'
    final_df['time'] = pd.to_datetime(final_df['time']).dt.strftime('%Y-%m')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    logger.info(f"Finished writing output to {out_path}")
