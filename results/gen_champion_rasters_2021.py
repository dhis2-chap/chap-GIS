"""Generate the 7 exposure layers for RWA 2021 using the CHAMPION suitability
curve (logistic threshold S(T)=1/(1+exp(-3*(T-23)))) instead of the Gaussian.

Spatial config is the optimum: lambda=1500, gamma=100, water buffer=2, 30 m.
The pipeline's run() hardcodes the Gaussian TPC, so suitability and exposure
are computed directly via compute_distance_field / exposure_from_field — the
same path exposure() uses internally, with the custom curve substituted.
"""
import json
from pathlib import Path
import numpy as np
import xarray as xr

import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.pipelines.malaria_exposure import reproject_layers, MalariaExposureParams

COUNTRY, LEVEL, YEAR = "RWA", 5, 2021
LAMBDA_M, GAMMA_M, WATER_BUF, RES = 1500.0, 100.0, 2, 30.0
champion = lambda T: 1.0 / (1.0 + np.exp(-3.0 * (T - 23.0)))   # logistic(23, k=3)
outdir = Path("results/champion_rasters_2021"); outdir.mkdir(parents=True, exist_ok=True)

params = MalariaExposureParams(resolution_m=RES, water_edge_buffer_pixels=WATER_BUF)
gdf = prepare_boundaries(COUNTRY, LEVEL)
aoi = cgis.aoi.buffered(gdf, params.aoi_buffer_deg)

land = chunk(cgis.io.worldcover.load(aoi=aoi, start=2021, end=2021, country_code=COUNTRY))
elev = chunk(cgis.io.elevation.load(aoi=aoi, country_code=COUNTRY))
rice = chunk(cgis.io.rice.load(country_code=COUNTRY))
pop = chunk(cgis.io.worldpop.load(country_code=COUNTRY, start=YEAR, end=YEAR))
pop.rio.write_crs("EPSG:4326", inplace=True)
tas = chunk(cgis.io.chelsa.load(gdf, start=f"{YEAR}-01", end=f"{YEAR}-12", country_code=COUNTRY))
for d in ("x", "y"):
    if d in tas.coords:
        tas[d] = np.round(tas[d].astype("float64"), 10)
tas.rio.write_crs("EPSG:4326", inplace=True)

print("reprojecting layers ...", flush=True)
L = reproject_layers(aoi=gdf, landcover_native=land, elev_native=elev,
                     tas_monthly=tas, population_native=pop, rice_native=rice, params=params)
crs = L.grid.rio.crs
tmpl = L.elev   # 2-D coords template

elev_np = np.asarray(L.elev.compute().values, np.float32)
temp_np = np.asarray(L.temperature.compute().values, np.float32)
land_np = np.asarray(cgis.landcover.land_mask(L.landcover).compute().values, bool)
water_np = np.asarray(cgis.landcover.water_mask(L.landcover).compute().values, bool)
breeding_np = np.asarray(cgis.landcover.breeding_site_mask(
    L.landcover, rice=L.rice_mask, water_edge_buffer=WATER_BUF).compute().values, bool)

print("champion suitability + exposure ...", flush=True)
suit_np = champion(temp_np).astype(np.float32)
suit_np[~np.isfinite(temp_np)] = np.nan
field = cgis.exposure.compute_distance_field(breeding_np, elev_np, pixel_m=RES,
            lambda_m=LAMBDA_M, land_mask=land_np, water_mask=water_np)
expo_np = cgis.exposure.exposure_from_field(field, suit_np, lambda_m=LAMBDA_M, gamma_m=GAMMA_M)

def da2d(arr, name, **attrs):
    return xr.DataArray(arr, dims=tmpl.dims, coords=tmpl.coords, name=name).rio.write_crs(crs).assign_attrs(attrs)

pop_da = L.population.rio.write_crs(crs)
expo_da = da2d(expo_np, "expo")
pop_exposure = (pop_da * expo_da).rename("pop_exposure").rio.write_crs(crs)

layers = {
    "breeding": da2d(breeding_np.astype("uint8"), "breeding"),
    "elev": da2d(elev_np, "elev"),
    "temperature": da2d(temp_np, "temperature"),
    "suitability": da2d(suit_np, "suitability", long_name="Logistic-threshold thermal suitability",
                        curve="logistic", T0=23.0, k=3.0),
    "population": pop_da,
    "expo": expo_da.assign_attrs(long_name="Exposure (champion logistic suitability)",
                                 horizontal_lambda_m=LAMBDA_M, vertical_gamma_m=GAMMA_M),
    "pop_exposure": pop_exposure,
}
print("computing + writing 7 layers ...", flush=True)
for name, da in layers.items():
    da = da.compute()
    p = outdir / f"{name}.nc"
    da.to_netcdf(p)
    print(f"wrote {p}  dims={dict(da.sizes)}", flush=True)

(outdir / "curve.json").write_text(json.dumps({
    "suitability_curve": "logistic threshold",
    "formula": "S(T) = 1 / (1 + exp(-3*(T-23)))", "T0_C": 23.0, "k": 3.0,
    "lambda_m": LAMBDA_M, "gamma_m": GAMMA_M, "water_edge_buffer_pixels": WATER_BUF,
    "resolution_m": RES, "year": YEAR,
    "note": "Champion curve from the suitability-curve search (rho_inc 0.515 vs Gaussian 0.496)."
}, indent=2))
print(f"DONE -> {outdir}", flush=True)
