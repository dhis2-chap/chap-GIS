"""Foundation feature table for the static-risk-map improvement steps.

Per-sector (pop-weighted means unless noted), on a 100 m grid, plus health data
and clean/raw targets and district + centroids. Lets each lever (spatial,
hydrology, urban, denominator) be a fast comparable evaluation. Saves
results/foundation_features.csv.
"""
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.special import expit
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import build_grid, reproject_to, reproject_population_to
from chap_gis.io import chelsa

RES, FK = 100.0, 1.0   # grid res; focal radius (km) for land-use features
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
aoi = cgis.aoi.buffered(gdf, 0.0027); grid = build_grid(gdf, resolution=RES / 111_000, crs="EPSG:4326")
print("loading rasters ...", flush=True)
land = chunk(cgis.io.worldcover.load(aoi=aoi, start=2021, end=2021, country_code="RWA"))
lc = np.asarray(reproject_to(land, grid, "mode").compute().values); lc = lc[0] if lc.ndim == 3 else lc
rice = cgis.io.rice.load(country_code="RWA")
if "band" in rice.dims: rice = rice.squeeze("band", drop=True)
rg = np.asarray(reproject_to(rice.astype("float32"), grid, "average").compute().values, np.float32); rg = rg[0] if rg.ndim == 3 else rg
gdf0 = cgis.io.boundaries.load("RWA", level=0)
tas = chelsa.load(gdf0, start="2021-01", end="2021-12", country_code="RWA").mean("time").rio.write_crs("EPSG:4326")
temp = np.asarray(reproject_to(tas, grid, "bilinear").compute().values, np.float32); temp = temp[0] if temp.ndim == 3 else temp
elev = chunk(cgis.io.elevation.load(aoi=aoi, country_code="RWA"))
zf = np.asarray(reproject_to(elev, grid, "bilinear").compute().values, np.float32); zf = zf[0] if zf.ndim == 3 else zf
wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True); wp.rio.write_crs("EPSG:4326", inplace=True)
pop = np.asarray(reproject_population_to(wp, grid, "bilinear").compute().values, np.float32); pop = pop[0] if pop.ndim == 3 else pop
pop = np.clip(np.nan_to_num(pop, nan=0.0), 0, None)

# hydrology from DEM
zc = ndimage.uniform_filter(np.nan_to_num(zf, nan=float(np.nanmean(zf))), size=11, mode="nearest")
gy_, gx_ = np.gradient(np.nan_to_num(zf, nan=float(np.nanmean(zf))), RES)   # m per m
slope = np.sqrt(gx_**2 + gy_**2)
valley = -(zf - zc)                       # >0 in local depressions (wetter)
flat = 1.0 / (slope + 0.05)               # flatness (water accumulates)
twi = np.log(flat * ndimage.uniform_filter(flat, size=11, mode="nearest") + 1)  # wetness proxy

def F(arr, km): return ndimage.uniform_filter(arr.astype(np.float32), size=int(round(km * 1000 / RES)) | 1, mode="nearest")
hab_pix = np.log1p(F(((rg > 0) | np.isin(lc, [90, 95])).astype(np.float32), FK))
blt_pix = np.log1p(F((lc == 50).astype(np.float32), FK))
wat_pix = np.log1p(F((lc == 80).astype(np.float32), FK))
wet_pix = np.log1p(F(np.isin(lc, [90, 95]).astype(np.float32), FK))
pdn_pix = np.log1p(F(pop, FK))

import rasterio.features as rfeat
sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)), out_shape=temp.shape,
        transform=grid.rio.transform(), fill=-1, dtype="int32")
ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(pop) & np.isfinite(zf)
s = sect[ok]; w = pop[ok].astype(np.float64); psum = np.bincount(s, weights=w, minlength=NS)
def sm(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)

cent = gdf.to_crs(4326).geometry.representative_point()
feat = pd.DataFrame({
    "location_id": loc, "wp_pop": psum,
    "temp": sm(temp), "sig_temp": sm(expit((temp - 19.0) / 0.5)),
    "elev": sm(zf), "slope": sm(slope), "valley": sm(valley), "twi": sm(twi),
    "hab": sm(hab_pix), "built": sm(blt_pix), "water": sm(wat_pix), "wetland": sm(wet_pix),
    "logpopdens": sm(pdn_pix),
    "lon": cent.x.to_numpy(), "lat": cent.y.to_numpy(),
})
# health + targets + district
sw = pd.read_csv("results/rwanda_sweep_temp.csv"); sw["location_id"] = sw.location_id.astype(str)
hd = sw.groupby("location_id").agg(cases=("disease", "sum"), dhis2_pop=("population", "mean"))
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
df = (feat.merge(hd, left_on="location_id", right_index=True)
          .merge(tgt.rename("inc_dhis2"), left_on="location_id", right_index=True)
          .merge(par.rename("parent"), left_on="location_id", right_index=True))
df["inc_wp"] = df["cases"] / df["wp_pop"].replace(0, np.nan) * 1000.0       # WorldPop-denominated incidence (cleaner)
cap99 = df["inc_wp"].quantile(0.99); df["inc_wp_w"] = df["inc_wp"].clip(upper=cap99)  # winsorised
df = df[(df.wp_pop > 0) & df.parent.notna() & df.inc_dhis2.notna()].reset_index(drop=True)
df.to_csv("results/foundation_features.csv", index=False)
print(f"wrote results/foundation_features.csv  n={len(df)}  cols={list(df.columns)}")
print(df[["temp", "elev", "slope", "valley", "twi", "hab", "built", "water", "logpopdens", "inc_dhis2", "inc_wp"]].describe().round(2).to_string())
