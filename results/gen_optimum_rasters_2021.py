"""Generate the 7 malaria-exposure raster layers for the optimum sweep config.

Optimum (best Spearman fit vs disease): lambda=1500, gamma=100, t_opt=29.
Last year in the RWA disease data = 2021.
"""
import json
from pathlib import Path

import numpy as np

import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.pipelines.malaria_exposure import run as run_pipeline, MalariaExposureParams

COUNTRY, LEVEL, YEAR = "RWA", 5, 2021
outdir = Path("results/optimum_rasters_2021")
outdir.mkdir(parents=True, exist_ok=True)

params = MalariaExposureParams(
    resolution_m=30.0,
    horizontal_lambda_m=1500.0,
    vertical_gamma_m=100.0,
    t_opt=29.0,
    t_sigma=6.0,
    t_min=19.0,
    t_max=38.0,
)

gdf = prepare_boundaries(COUNTRY, LEVEL)
aoi = cgis.aoi.buffered(gdf, params.aoi_buffer_deg)

wc_year = max(2020, min(YEAR, 2021))
land = chunk(cgis.io.worldcover.load(aoi=aoi, start=wc_year, end=wc_year, country_code=COUNTRY))
elev = chunk(cgis.io.elevation.load(aoi=aoi, country_code=COUNTRY))
rice = chunk(cgis.io.rice.load(country_code=COUNTRY))

pop = chunk(cgis.io.worldpop.load(country_code=COUNTRY, start=YEAR, end=YEAR))
pop.rio.write_crs("EPSG:4326", inplace=True)

tas = chunk(cgis.io.chelsa.load(gdf, start=f"{YEAR}-01", end=f"{YEAR}-12", country_code=COUNTRY))
for dim in ("x", "y"):
    if dim in tas.coords:
        tas[dim] = np.round(tas[dim].astype("float64"), 10)
tas.rio.write_crs("EPSG:4326", inplace=True)

print(f"Running pipeline for {COUNTRY} {YEAR} at optimum params ...", flush=True)
ds = run_pipeline(
    aoi=gdf,
    landcover_native=land,
    elev_native=elev,
    tas_monthly=tas,
    population_native=pop,
    rice_native=rice,
    params=params,
)
print("Computing 7 layers (this is the heavy step) ...", flush=True)
ds = ds.compute()

for v in ds.data_vars:
    p = outdir / f"{v}.nc"
    ds[v].to_netcdf(p)
    print(f"wrote {p}  dims={dict(ds[v].sizes)}", flush=True)

(outdir / "params.json").write_text(json.dumps(params.model_dump(), indent=2))
print(f"DONE -> {outdir}  ({len(ds.data_vars)} layers)", flush=True)
