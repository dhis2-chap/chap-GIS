"""Monthly-resolved suitability-evaluation cache.

Computes thermal suitability inputs per MONTH (not annual mean), so different
monthly->yearly aggregations and season-aligned year windows can be scored
against disease cheaply.

For each month we resample that month's downscaled temperature at the FIXED
nearest-breeding-site indices (the distance field is temperature-independent,
so it is built once). We then accumulate, per sector:

  W[sector, month, bin]   = sum_pixels pop*base binned by T_nearest(month)
        -> supports mean / sum / threshold-count aggregation over ANY window
           (calendar Jan-Dec or season-aligned Sep-Aug), for any curve S.
  Wmax_cal[sector, y, bin]   binned by per-pixel max-over-Jan-Dec  T_nearest
  Wmax_seas[sector, sy, bin] binned by per-pixel max-over-Sep-Aug  T_nearest
        -> warmest-month ("max") aggregation, exact for monotone S.

Fixed optimum spatial config: lambda=1500, gamma=100, water buffer=2.
"""
import numpy as np
import rasterio.features as rfeat

import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import build_grid, reproject_to, reproject_population_to

COUNTRY, LEVEL = "RWA", 5
YEARS = list(range(2013, 2022))
LAMBDA_M, GAMMA_M, WATER_BUF, RES = 1500.0, 100.0, 2, 30.0
BIN_LO, BIN_HI, BIN_W = 5.0, 42.0, 0.25
edges = np.arange(BIN_LO, BIN_HI + BIN_W, BIN_W)
centers = (edges[:-1] + edges[1:]) / 2
NB = len(centers)

def binidx(t):
    return np.clip(((t - BIN_LO) / BIN_W).astype(np.int64), 0, NB - 1)

gdf = prepare_boundaries(COUNTRY, LEVEL)
loc_ids = gdf["location_id"].to_numpy()
NS = len(gdf)
aoi = cgis.aoi.buffered(gdf, 0.0027)

land = chunk(cgis.io.worldcover.load(aoi=aoi, start=2021, end=2021, country_code=COUNTRY))
elev_n = chunk(cgis.io.elevation.load(aoi=aoi, country_code=COUNTRY))
rice = chunk(cgis.io.rice.load(country_code=COUNTRY))
popall = chunk(cgis.io.worldpop.load(country_code=COUNTRY, start=min(YEARS), end=max(YEARS)))
popall.rio.write_crs("EPSG:4326", inplace=True)

# ---- static grid + distance field (temperature-independent) ----
print("building grid + distance field (once) ...", flush=True)
grid = build_grid(gdf, resolution=RES / 111_000, crs="EPSG:4326")
landcover = reproject_to(land, grid, "mode").astype("uint8")
elev_g_da = reproject_to(elev_n, grid, "bilinear")
elev_g = np.asarray(elev_g_da.compute().values, np.float32)
rice_mask = (reproject_to(rice, grid, "average") > 0).rio.write_crs(grid.rio.crs)
land_np = np.asarray(cgis.landcover.land_mask(landcover).compute().values, bool)
water_np = np.asarray(cgis.landcover.water_mask(landcover).compute().values, bool)
breeding = np.asarray(cgis.landcover.breeding_site_mask(
    landcover, rice=rice_mask, water_edge_buffer=WATER_BUF).compute().values, bool)
field = cgis.exposure.compute_distance_field(
    breeding, elev_g, pixel_m=RES, lambda_m=LAMBDA_M, land_mask=land_np, water_mask=water_np)
base = cgis.exposure.exposure_from_field(field, None, lambda_m=LAMBDA_M, gamma_m=GAMMA_M)
basefin = np.isfinite(base)
v = field.valid

sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
                       out_shape=base.shape, transform=grid.rio.transform(),
                       fill=-1, dtype="int32")

# coarse-elevation template for lapse downscaling (CHELSA grid is constant)
tas0 = chunk(cgis.io.chelsa.load(gdf, start="2013-01", end="2013-01", country_code=COUNTRY))
tas0.rio.write_crs("EPSG:4326", inplace=True)
coarse_elev_g = (elev_n.pipe(reproject_to, tas0, "average").pipe(reproject_to, grid, "bilinear"))

def temp_month(tas_m):
    on_grid = reproject_to(tas_m, grid, "bilinear")
    return np.asarray(cgis.climate.lapse_rate_downscale(on_grid, coarse_elev_g, elev_g_da)
                      .compute().values, np.float32)

NM = len(YEARS) * 12
W = np.zeros((NS, NM, NB), np.float32)
SPOP = np.zeros((NS, len(YEARS)), np.float64)
Wmax_cal = np.zeros((NS, len(YEARS), NB), np.float32)
season_starts = list(range(min(YEARS), max(YEARS)))   # Sep(y)->Aug(y+1)
Wmax_seas = np.zeros((NS, len(season_starts), NB), np.float32)
months_meta = []

cal_max = np.full(base.shape, np.nan, np.float32)
seas_max = np.full(base.shape, np.nan, np.float32)
seas_open = False
seas_idx = -1

mi = 0
for yi, y in enumerate(YEARS):
    pop_y = np.asarray(reproject_population_to(
        popall.sel(time=f"{y}").squeeze(drop=True), grid, "bilinear").compute().values, np.float32)
    if pop_y.ndim == 3:
        pop_y = pop_y[0]
    weight = pop_y * base                      # NaN where base NaN
    okp = (sect >= 0) & np.isfinite(pop_y)
    SPOP[:, yi] = np.bincount(sect[okp], weights=pop_y[okp].astype(np.float64), minlength=NS)

    tas_y = chunk(cgis.io.chelsa.load(gdf, start=f"{y}-01", end=f"{y}-12", country_code=COUNTRY))
    for d in ("x", "y"):
        if d in tas_y.coords:
            tas_y[d] = np.round(tas_y[d].astype("float64"), 10)
    tas_y.rio.write_crs("EPSG:4326", inplace=True)
    print(f"[{y}] 12 monthly temps ...", flush=True)
    for m in range(1, 13):
        tnow = temp_month(tas_y.isel(time=m - 1))
        tnear = np.full(base.shape, np.nan, np.float32)
        tnear[v] = tnow[field.iy[v], field.ix[v]]
        # monthly histogram (weight pop*base)
        ok = (sect >= 0) & basefin & np.isfinite(tnear)
        W[:, mi, :] = np.bincount(sect[ok] * NB + binidx(tnear[ok]),
                                  weights=weight[ok].astype(np.float64),
                                  minlength=NS * NB).reshape(NS, NB)
        months_meta.append((y, m))
        # running maxes (per pixel)
        cal_max = np.fmax(cal_max, tnear)
        if m == 9:
            if y in season_starts:                       # no Sep window in the final year
                seas_max = tnear.copy(); seas_open = True; seas_idx = season_starts.index(y)
            else:
                seas_open = False
        elif seas_open:
            seas_max = np.fmax(seas_max, tnear)
        if m == 12:
            okc = (sect >= 0) & basefin & np.isfinite(cal_max)
            Wmax_cal[:, yi, :] = np.bincount(sect[okc] * NB + binidx(cal_max[okc]),
                                             weights=weight[okc].astype(np.float64),
                                             minlength=NS * NB).reshape(NS, NB)
            cal_max[:] = np.nan
        if m == 8 and seas_open:
            oks = (sect >= 0) & basefin & np.isfinite(seas_max)
            Wmax_seas[:, seas_idx, :] = np.bincount(sect[oks] * NB + binidx(seas_max[oks]),
                                                    weights=weight[oks].astype(np.float64),
                                                    minlength=NS * NB).reshape(NS, NB)
            seas_open = False
        mi += 1

np.savez_compressed(
    "results/monthly_suit_cache.npz",
    W=W, sector_pop=SPOP, Wmax_cal=Wmax_cal, Wmax_seas=Wmax_seas,
    bin_centers=centers, location_ids=loc_ids.astype("U"),
    months=np.array(months_meta), years=np.array(YEARS),
    season_starts=np.array(season_starts),
    lambda_m=LAMBDA_M, gamma_m=GAMMA_M, water_buffer=WATER_BUF)
print("WROTE results/monthly_suit_cache.npz", W.shape, flush=True)
