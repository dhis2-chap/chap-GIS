"""Softmax (smooth-max) monthly->yearly aggregation at several intensities.

For the fixed winning curve S=logistic(23,k3), per pixel and calendar year:
  S_year = sum_m S_m * exp(beta*S_m) / sum_m exp(beta*S_m)
beta=0 -> mean (known 0.5162), beta->inf -> max (known 0.5187).
Then risk = sum_pixels pop*base*S_year / sector_pop ; Spearman vs incidence.

Needs a fresh pass over the 108 monthly temperatures (softmax is a per-pixel
cross-month nonlinearity, not reconstructable from the binned caches).
"""
import numpy as np
import pandas as pd
import rasterio.features as rfeat

import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import build_grid, reproject_to, reproject_population_to

COUNTRY, LEVEL = "RWA", 5
YEARS = list(range(2013, 2022))
LAMBDA_M, GAMMA_M, WATER_BUF, RES = 1500.0, 100.0, 2, 30.0
BETAS = [0.0, 1.0, 3.0, 8.0, 20.0, 50.0]      # 0=mean ... large=max
S = lambda x: 1.0 / (1.0 + np.exp(-3.0 * (x - 23.0)))   # logistic(23,k3)

gdf = prepare_boundaries(COUNTRY, LEVEL)
loc = gdf["location_id"].to_numpy(); NS = len(gdf)
aoi = cgis.aoi.buffered(gdf, 0.0027)
land = chunk(cgis.io.worldcover.load(aoi=aoi, start=2021, end=2021, country_code=COUNTRY))
elev_n = chunk(cgis.io.elevation.load(aoi=aoi, country_code=COUNTRY))
rice = chunk(cgis.io.rice.load(country_code=COUNTRY))
popall = chunk(cgis.io.worldpop.load(country_code=COUNTRY, start=min(YEARS), end=max(YEARS)))
popall.rio.write_crs("EPSG:4326", inplace=True)

print("static grid + distance field ...", flush=True)
grid = build_grid(gdf, resolution=RES / 111_000, crs="EPSG:4326")
landcover = reproject_to(land, grid, "mode").astype("uint8")
elev_da = reproject_to(elev_n, grid, "bilinear")
elev_g = np.asarray(elev_da.compute().values, np.float32)
rice_mask = (reproject_to(rice, grid, "average") > 0).rio.write_crs(grid.rio.crs)
land_np = np.asarray(cgis.landcover.land_mask(landcover).compute().values, bool)
water_np = np.asarray(cgis.landcover.water_mask(landcover).compute().values, bool)
breeding = np.asarray(cgis.landcover.breeding_site_mask(
    landcover, rice=rice_mask, water_edge_buffer=WATER_BUF).compute().values, bool)
field = cgis.exposure.compute_distance_field(breeding, elev_g, pixel_m=RES,
            lambda_m=LAMBDA_M, land_mask=land_np, water_mask=water_np)
base = cgis.exposure.exposure_from_field(field, None, lambda_m=LAMBDA_M, gamma_m=GAMMA_M)
basefin = np.isfinite(base); v = field.valid
sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=base.shape, transform=grid.rio.transform(), fill=-1, dtype="int32")
tas0 = chunk(cgis.io.chelsa.load(gdf, start="2013-01", end="2013-01", country_code=COUNTRY))
tas0.rio.write_crs("EPSG:4326", inplace=True)
coarse_elev_g = elev_n.pipe(reproject_to, tas0, "average").pipe(reproject_to, grid, "bilinear")

def temp_month(tas_m):
    og = reproject_to(tas_m, grid, "bilinear")
    return np.asarray(cgis.climate.lapse_rate_downscale(og, coarse_elev_g, elev_da).compute().values, np.float32)

NB = len(BETAS)
PE = np.zeros((NS, len(YEARS), NB), np.float64)
SPOP = np.zeros((NS, len(YEARS)), np.float64)
shp = base.shape

for yi, y in enumerate(YEARS):
    pop_y = np.asarray(reproject_population_to(
        popall.sel(time=f"{y}").squeeze(drop=True), grid, "bilinear").compute().values, np.float32)
    if pop_y.ndim == 3: pop_y = pop_y[0]
    weight = (pop_y * base).astype(np.float64)
    okp = (sect >= 0) & np.isfinite(pop_y)
    SPOP[:, yi] = np.bincount(sect[okp], weights=pop_y[okp].astype(np.float64), minlength=NS)

    num = [np.zeros(shp, np.float64) for _ in BETAS]   # sum S*exp(bS)
    den = [np.zeros(shp, np.float64) for _ in BETAS]   # sum exp(bS)
    tas_y = chunk(cgis.io.chelsa.load(gdf, start=f"{y}-01", end=f"{y}-12", country_code=COUNTRY))
    for d in ("x", "y"):
        if d in tas_y.coords: tas_y[d] = np.round(tas_y[d].astype("float64"), 10)
    tas_y.rio.write_crs("EPSG:4326", inplace=True)
    print(f"[{y}] 12 monthly temps ...", flush=True)
    for m in range(12):
        t = temp_month(tas_y.isel(time=m))
        sm = np.zeros(shp, np.float32)
        sm[v] = S(t[field.iy[v], field.ix[v]]).astype(np.float32)   # monthly suitability per pixel
        for bi, b in enumerate(BETAS):
            e = np.exp(b * sm)
            den[bi] += e
            num[bi] += sm * e
    for bi in range(NB):
        sy = np.divide(num[bi], den[bi], out=np.zeros(shp), where=den[bi] > 0)  # softmax per pixel
        w = weight * sy
        ok = (sect >= 0) & basefin & np.isfinite(w)
        PE[:, yi, bi] = np.bincount(sect[ok], weights=w[ok], minlength=NS)

np.savez_compressed("results/softmax_pe.npz", PE=PE, sector_pop=SPOP,
                    betas=np.array(BETAS), location_ids=loc.astype("U"), years=np.array(YEARS))

# ---- score vs disease (calendar) ----
dis = pd.read_csv("rwanda_spray.csv")[["location_id", "time", "disease", "population"]].copy()
dis["key"] = pd.to_datetime(dis["time"]).dt.year
dis = dis[dis.population > 0]
dd = dis.groupby(["location_id", "key"]).agg(disease=("disease", "sum"), pop=("population", "mean")).reset_index()

print(f"\n{'beta':>6}  {'regime':10}{'rho_inc':>9}{'rho_raw':>9}")
for bi, b in enumerate(BETAS):
    risk = np.divide(PE[:, :, bi], SPOP, out=np.zeros_like(PE[:, :, bi]), where=SPOP > 0)
    rdf = pd.DataFrame({"location_id": np.repeat(loc, len(YEARS)),
                        "key": np.tile(YEARS, len(loc)), "risk": risk.ravel()})
    m = dd.merge(rdf, on=["location_id", "key"]); m = m[m["pop"] > 0]
    ri = m["risk"].corr(m["disease"] / m["pop"], method="spearman")
    rr = m["risk"].corr(m["disease"], method="spearman")
    reg = "= mean" if b == 0 else ("~ max" if b >= 50 else "")
    print(f"{b:>6.0f}  {reg:10}{ri:>9.4f}{rr:>9.4f}")
print("ref: mean=0.5162  max(warmest-month)=0.5187")
