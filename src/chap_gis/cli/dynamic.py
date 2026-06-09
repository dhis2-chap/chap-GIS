from pathlib import Path
import json
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
    ExposureSweepSpec,
    combo_tag,
    reproject_layers,
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
    id_col = next(
        (c for c in ("shapeID", "shapeName") if c in gdf.columns),
        None,
    )
    gdf["location_id"] = (gdf[id_col] if id_col else gdf.index).astype(str)
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
    # Keep only the disease signal — extra CSV columns (e.g. climate covariates,
    # or a prior run's tas/population/pop_exposure) would otherwise ride into the
    # final xr.merge and collide with the freshly computed regional aggregates.
    df = df[["location_id", "time", "disease"]]
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
            logger.warning(
                "No temperature data for %s — skipping year. "
                "Disease/population rows for this year will not appear in the output.",
                y,
            )
            continue
        if tas_y.time.size < 12:
            logger.warning(
                "Only %d of 12 months of temperature data for %s — "
                "annual mean will be biased toward the available months.",
                tas_y.time.size, y,
            )

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
        logger.info(f"Aggregating exposure for year {y}... to regions")
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


def _as_named_dataset(agg, name: str) -> xr.Dataset:
    """Coerce an ``aggregate_to_regions`` result to a Dataset with one var ``name``."""
    if isinstance(agg, xr.DataArray):
        return agg.to_dataset(name=name)
    var = next(iter(agg.data_vars))
    return agg.rename({var: name})


def _load_sweep_spec(path: Path) -> ExposureSweepSpec:
    """Read a YAML/JSON exposure parameter grid into an :class:`ExposureSweepSpec`."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "YAML grid config needs PyYAML (`pip install pyyaml`); "
                "alternatively pass a .json file."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return ExposureSweepSpec.model_validate(data)


def _run_sweep_logic(gdf, health, pop, tas, country, spec: ExposureSweepSpec):
    """Multi-combo variant of :func:`_run_core_logic`.

    Reprojects inputs once per year, computes the distance field once per
    ``water_edge_buffer_pixels`` value, then applies the cheap thermal /
    lambda / gamma kernels — so N combos cost N region-aggregations but only
    one reprojection per year and one distance transform per water buffer.
    Returns a region-level dataset with one ``pop_exposure__<tag>`` var per
    combo, or ``None`` if no year produced data.
    """
    years = np.unique(health.time.dt.year.values)
    combos = spec.combos()
    nt, nl, ng = len(spec.thermal), len(spec.lambda_m), len(spec.gamma_m)
    max_lambda = max(spec.lambda_m)
    res = spec.base.resolution_m

    worldcover_year = max(2020, min(int(health.time.dt.year.max()), 2021))
    aoi = cgis.aoi.buffered(gdf, spec.base.aoi_buffer_deg)

    land = chunk(cgis.io.worldcover.load(aoi=aoi, start=worldcover_year, end=worldcover_year, country_code=country))
    elev = chunk(cgis.io.elevation.load(aoi=aoi, country_code=country))
    rice = chunk(cgis.io.rice.load(country_code=country))

    logger.info(
        "Exposure sweep: %d combos over %d year(s); "
        "%d distance-field build(s) and %d kernel(s) per year",
        len(combos), len(years), len(spec.water_edge_buffer_pixels), len(combos),
    )

    per_tag: dict[str, list] = {c.tag: [] for c in combos}

    for y in tqdm(years, desc=f"Sweep years for {country}"):
        tas_y = tas.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
        pop_y = pop.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))

        if tas_y.time.size == 0:
            logger.warning("No temperature data for %s — skipping year.", y)
            continue
        if tas_y.time.size < 12:
            logger.warning(
                "Only %d of 12 months of temperature data for %s — "
                "annual mean will be biased toward the available months.",
                tas_y.time.size, y,
            )

        layers = reproject_layers(
            aoi=gdf,
            landcover_native=land,
            elev_native=elev,
            tas_monthly=tas_y,
            population_native=pop_y,
            rice_native=rice,
            params=spec.base,
        )

        # Materialise the per-year shared inputs once. Population/temperature
        # are persisted (kept lazy but cached) so each combo's region
        # aggregation reuses them instead of recomputing the reprojection.
        elev_np = np.asarray(layers.elev.compute().values, dtype=np.float32)
        land_np = np.asarray(cgis.landcover.land_mask(layers.landcover).compute().values, dtype=bool)
        water_np = np.asarray(cgis.landcover.water_mask(layers.landcover).compute().values, dtype=bool)
        pop_p = layers.population.persist()
        temp_p = layers.temperature.persist()
        crs = layers.grid.rio.crs
        template = layers.elev  # 2-D dims/coords for wrapping kernel output

        for wb_i, wb in enumerate(spec.water_edge_buffer_pixels):
            breeding_np = np.asarray(
                cgis.landcover.breeding_site_mask(
                    layers.landcover, rice=layers.rice_mask, water_edge_buffer=wb
                ).compute().values,
                dtype=bool,
            )
            field = cgis.exposure.compute_distance_field(
                breeding_np, elev_np, pixel_m=res, lambda_m=max_lambda,
                land_mask=land_np, water_mask=water_np,
            )

            for th_i, th in enumerate(spec.thermal):
                suit_np = np.asarray(
                    cgis.suitability.thermal_suitability(
                        temp_p, t_opt=th.t_opt, sigma=th.sigma,
                        t_min=th.t_min, t_max=th.t_max,
                    ).compute().values,
                    dtype=np.float32,
                )

                for lam_i, lam in enumerate(spec.lambda_m):
                    for gam_i, gam in enumerate(spec.gamma_m):
                        idx = ((wb_i * nt + th_i) * nl + lam_i) * ng + gam_i
                        tag = combo_tag(idx)

                        expo_np = cgis.exposure.exposure_from_field(
                            field, suit_np, lambda_m=lam, gamma_m=gam
                        )
                        expo_da = xr.DataArray(
                            expo_np, dims=template.dims, coords=template.coords
                        ).rio.write_crs(crs)
                        pop_exposure = (pop_p * expo_da).rename(
                            f"pop_exposure__{tag}"
                        ).rio.write_crs(crs)

                        agg = aggregate_to_regions(
                            pop_exposure, gdf, statistic="sum", id_field="location_id"
                        )
                        agg_y = _as_named_dataset(agg, f"pop_exposure__{tag}")
                        agg_y = agg_y.assign_coords(time=pop_y.time)
                        per_tag[tag].append(agg_y)

    merged = [xr.concat(v, dim="time") for v in per_tag.values() if v]
    return xr.merge(merged, join="inner") if merged else None


def dynamic_periods(
    country: str,
    level: int = 5,
    inter: bool = True,
    input_csv: str = "./data/inputs/disease-data.csv",
    out_path: Path = Path("test.csv"),
    grid_config: Path | None = None,
):
    """Multi-year, multi-month malaria exposure pipeline with health data.

    With ``grid_config`` set (a YAML/JSON :class:`ExposureSweepSpec`), runs a
    parameter sweep and writes one ``pop_exposure__<tag>`` /
    ``mean_exposure_per_person__<tag>`` column pair per combo, plus a
    ``<out>.params.json`` manifest mapping each tag to its parameters.
    """
    logger.info(f"Starting pipeline for {country}")

    gdf = prepare_boundaries(country, level)
    health = get_health_data(input_csv, gdf)
    pop, pop_agg, tas, tas_agg = get_environmental_data(country, health, gdf, inter)

    if grid_config is None:
        params = MalariaExposureParams(resolution_m=30.0)
        expo = _run_core_logic(gdf, health, pop, tas, country, params)

        logger.info("Merging regional datasets...")
        # These are already aggregated to regions (small), so merge is safe
        datasets_to_merge = [ds for ds in [health, tas_agg, pop_agg, expo] if ds is not None]
        final = xr.merge(datasets_to_merge, join="inner")

        # Population-weighted mean exposure index per person in each region:
        # Σ(pop·expo) / Σ(pop); guard regions with zero population.
        final["mean_exposure_per_person"] = (
            final["pop_exposure"] / final["population"].where(final["population"] > 0)
        )

        # Final export to pandas (Safe only because these are region-level aggregates)
        df = final.to_dataframe().reset_index()

        if not df.empty:
            df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m")
            cols_to_keep = ["location_id", "time", "disease", "tas", "population", "pop_exposure", "mean_exposure_per_person"]
            df = df[[c for c in cols_to_keep if c in df.columns]]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Done. Rows written: {len(df)}")
        return

    # --- Parameter-sweep path -------------------------------------------------
    spec = _load_sweep_spec(grid_config)
    combos = spec.combos()
    logger.info("Loaded sweep grid with %d parameter combinations", len(combos))

    expo = _run_sweep_logic(gdf, health, pop, tas, country, spec)

    logger.info("Merging regional datasets...")
    datasets_to_merge = [ds for ds in [health, tas_agg, pop_agg, expo] if ds is not None]
    final = xr.merge(datasets_to_merge, join="inner")

    pop_safe = final["population"].where(final["population"] > 0)
    expo_cols: list[str] = []
    for c in combos:
        pe = f"pop_exposure__{c.tag}"
        if pe not in final:
            continue
        mepp = f"mean_exposure_per_person__{c.tag}"
        final[mepp] = final[pe] / pop_safe
        expo_cols += [pe, mepp]

    df = final.to_dataframe().reset_index()
    base_cols = ["location_id", "time", "disease", "tas", "population"]
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m")
        df = df[[col for col in base_cols + expo_cols if col in df.columns]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    manifest_path = out_path.with_suffix(".params.json")
    manifest = {
        "country": country,
        "level": level,
        "columns": {
            c.tag: {
                "pop_exposure_column": f"pop_exposure__{c.tag}",
                "mean_exposure_per_person_column": f"mean_exposure_per_person__{c.tag}",
                "lambda_m": c.lambda_m,
                "gamma_m": c.gamma_m,
                "water_edge_buffer_pixels": c.water_edge_buffer_pixels,
                "t_opt": c.thermal.t_opt,
                "sigma": c.thermal.sigma,
                "t_min": c.thermal.t_min,
                "t_max": c.thermal.t_max,
            }
            for c in combos
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Done. Rows written: %d across %d combos. Params manifest: %s",
        len(df), len(combos), manifest_path,
    )