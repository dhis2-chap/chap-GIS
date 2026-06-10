"""Cache per-sector (population x distance-kernel) weights binned by nearest
breeding-site temperature, so arbitrary thermal-suitability curves can be
scored against disease in milliseconds.

Identity exploited:  exposure = base(lambda,gamma,breeding) * S(T_nearest)
  base       = exp(-d/lambda) * exp(-max(dz,0)/gamma)   (suitability = None)
  T_nearest  = temperature at the nearest breeding pixel
So  sector_pop_exposure(S) = sum_bin S(T_bin) * W[sector, bin]
with W[sector, bin] = sum_{pixels in sector, T_nearest in bin} population * base.

Fixed at the optimum spatial config: lambda=1500, gamma=100, water buffer=2.
"""
import numpy as np
import rasterio.features as rfeat

import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.pipelines.malaria_exposure import reproject_layers, MalariaExposureParams

COUNTRY, LEVEL = "RWA", 5
YEARS = list(range(2013, 2022))
LAMBDA_M, GAMMA_M, WATER_BUF = 1500.0, 100.0, 2
RES = 30.0

# temperature bins for T_nearest (breeding-site temps); fine grid, cheap to store
BIN_LO, BIN_HI, BIN_W = 5.0, 42.0, 0.25
edges = np.arange(BIN_LO, BIN_HI + BIN_W, BIN_W)
centers = (edges[:-1] + edges[1:]) / 2
NB = len(centers)

params = MalariaExposureParams(resolution_m=RES)  # geometry only; thermal unused here

gdf = prepare_boundaries(COUNTRY, LEVEL)
loc_ids = gdf["location_id"].to_numpy()
NS = len(gdf)
aoi = cgis.aoi.buffered(gdf, params.aoi_buffer_deg)

wc_year = 2021
land = chunk(cgis.io.worldcover.load(aoi=aoi, start=wc_year, end=wc_year, country_code=COUNTRY))
elev = chunk(cgis.io.elevation.load(aoi=aoi, country_code=COUNTRY))
rice = chunk(cgis.io.rice.load(country_code=COUNTRY))
popall = chunk(cgis.io.worldpop.load(country_code=COUNTRY, start=min(YEARS), end=max(YEARS)))
popall.rio.write_crs("EPSG:4326", inplace=True)

W = np.zeros((NS, len(YEARS), NB), dtype=np.float64)
SPOP = np.zeros((NS, len(YEARS)), dtype=np.float64)

for yi, y in enumerate(YEARS):
    print(f"[{y}] reprojecting + distance field ...", flush=True)
    tas = chunk(cgis.io.chelsa.load(gdf, start=f"{y}-01", end=f"{y}-12", country_code=COUNTRY))
    for d in ("x", "y"):
        if d in tas.coords:
            tas[d] = np.round(tas[d].astype("float64"), 10)
    tas.rio.write_crs("EPSG:4326", inplace=True)
    pop_y = popall.sel(time=f"{y}").squeeze(drop=True)

    L = reproject_layers(aoi=gdf, landcover_native=land, elev_native=elev,
                         tas_monthly=tas, population_native=pop_y,
                         rice_native=rice, params=params)
    elev_np = np.asarray(L.elev.compute().values, np.float32)
    temp_np = np.asarray(L.temperature.compute().values, np.float32)
    pop_np = np.asarray(L.population.compute().values, np.float32)
    if pop_np.ndim == 3:
        pop_np = pop_np[0]
    land_np = np.asarray(cgis.landcover.land_mask(L.landcover).compute().values, bool)
    water_np = np.asarray(cgis.landcover.water_mask(L.landcover).compute().values, bool)
    breeding = np.asarray(cgis.landcover.breeding_site_mask(
        L.landcover, rice=L.rice_mask, water_edge_buffer=WATER_BUF).compute().values, bool)

    field = cgis.exposure.compute_distance_field(
        breeding, elev_np, pixel_m=RES, lambda_m=LAMBDA_M,
        land_mask=land_np, water_mask=water_np)
    base = cgis.exposure.exposure_from_field(field, None, lambda_m=LAMBDA_M, gamma_m=GAMMA_M)

    # nearest-breeding-site temperature per pixel
    tnear = np.full(base.shape, np.nan, np.float32)
    v = field.valid
    tnear[v] = temp_np[field.iy[v], field.ix[v]]

    # sector index raster (once; grid is identical across years)
    if yi == 0:
        transform = L.grid.rio.transform()
        sect = rfeat.rasterize(
            ((geom, i) for i, geom in enumerate(gdf.geometry)),
            out_shape=base.shape, transform=transform, fill=-1, dtype="int32")

    weight = pop_np * base  # NaN where base is NaN (water/non-land)
    ok = (sect >= 0) & np.isfinite(weight) & np.isfinite(tnear)
    s = sect[ok]
    b = np.clip(((tnear[ok] - BIN_LO) / BIN_W).astype(np.int64), 0, NB - 1)
    w = weight[ok].astype(np.float64)
    flat = np.bincount(s * NB + b, weights=w, minlength=NS * NB).reshape(NS, NB)
    W[:, yi, :] = flat

    okp = (sect >= 0) & np.isfinite(pop_np)
    SPOP[:, yi] = np.bincount(sect[okp], weights=pop_np[okp].astype(np.float64), minlength=NS)
    print(f"[{y}] done. nonzero sector-bins={int((flat>0).sum())}", flush=True)

np.savez_compressed(
    "results/suit_eval_cache.npz",
    W=W, sector_pop=SPOP, bin_centers=centers,
    location_ids=loc_ids.astype("U"), years=np.array(YEARS),
    lambda_m=LAMBDA_M, gamma_m=GAMMA_M, water_buffer=WATER_BUF)
print("WROTE results/suit_eval_cache.npz", W.shape, flush=True)
