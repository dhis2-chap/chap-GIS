"""Aggregate raster terrain/habitat layers to per-sector covariates:
elevation (mean), rice fraction, wetland/marshland fraction, water fraction.
Then re-run the multivariate models with these added to the env set.
"""
import numpy as np, pandas as pd
import rasterio.features as rfeat
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr

import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import build_grid, reproject_to

RES = 100.0  # coarse grid is plenty for per-sector means; ~10x faster than 30 m
gdf = prepare_boundaries("RWA", 5)
loc = gdf["location_id"].to_numpy(); NS = len(gdf)
aoi = cgis.aoi.buffered(gdf, 0.0027)

land = chunk(cgis.io.worldcover.load(aoi=aoi, start=2021, end=2021, country_code="RWA"))
elev = chunk(cgis.io.elevation.load(aoi=aoi, country_code="RWA"))
rice = cgis.io.rice.load(country_code="RWA")
if "band" in rice.dims: rice = rice.squeeze("band", drop=True)

print("reprojecting terrain layers to grid ...", flush=True)
grid = build_grid(gdf, resolution=RES/111_000, crs="EPSG:4326")
elev_g = np.asarray(reproject_to(elev, grid, "bilinear").compute().values, np.float32)
lc_g   = np.asarray(reproject_to(land, grid, "mode").compute().values)
rice_g = np.asarray(reproject_to(rice.astype("float32"), grid, "average").compute().values, np.float32)
if rice_g.ndim == 3: rice_g = rice_g[0]

wet  = ((lc_g == 90) | (lc_g == 95)).astype(np.float32)   # WorldCover herbaceous wetland + mangrove
water= (lc_g == 80).astype(np.float32)                    # permanent water
rice_g = np.where(np.isfinite(rice_g), rice_g > 0, 0).astype(np.float32)  # rice presence fraction

sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=elev_g.shape, transform=grid.rio.transform(), fill=-1, dtype="int32")
ok = sect >= 0
cnt = np.bincount(sect[ok], minlength=NS).astype(float)
def per_sector(arr, valid):
    m = ok & valid
    return np.bincount(sect[m], weights=arr[m].astype(float), minlength=NS) / np.maximum(cnt, 1)

terr = pd.DataFrame({
    "location_id": loc,
    "elevation_m":   per_sector(elev_g, np.isfinite(elev_g)),
    "rice_frac":     per_sector(rice_g, np.isfinite(rice_g)),
    "wetland_frac":  per_sector(wet, np.isfinite(lc_g)),
    "water_frac":    per_sector(water, np.isfinite(lc_g)),
}).set_index("location_id")
terr.to_csv("results/rwanda_sector_terrain.csv")
print("\nper-sector terrain covariates (head):")
print(terr.describe().round(3).to_string())

# ---- modeling ----
cov = pd.read_csv("data/inputs/chap_data_level5_clean_2013-2021.csv").rename(columns={"location":"location_id"})
cov["location_id"] = cov["location_id"].astype(str)
ENV = ["mean_temperature","max_temperature","min_temperature","dewpoint_temperature",
       "relative_humidity","rainfall_era5","rainfall_chirps","rainfall_iri","evi","ndvi"]
sec = cov.groupby("location_id")[ENV].mean()
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")
sec = sec.join(tgt["annual_incidence_per1000"]).join(terr)
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID","parent"]]; g["location_id"]=g.shapeID.astype(str)
sec = sec.join(g.set_index("location_id")["parent"]).dropna(subset=["parent","annual_incidence_per1000"])
y = sec["annual_incidence_per1000"].values
groups = sec["parent"].astype(str).values
TERR = ["elevation_m","rice_frac","wetland_frac","water_frac"]

def lodo(model, cols):
    X = sec[cols].fillna(sec[cols].mean()).values
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(pred, y).statistic

MODELS = {"Linear":LinearRegression(),
          "RandomForest":RandomForestRegressor(n_estimators=600,min_samples_leaf=3,random_state=0,n_jobs=-1),
          "GradBoost":GradientBoostingRegressor(n_estimators=400,max_depth=3,learning_rate=0.03,subsample=0.8,random_state=0)}
SETS = {"terrain only":TERR, "ENV (10)":ENV, "ENV + terrain":ENV+TERR}
print(f"\n=== Leave-one-district-out Spearman vs incidence (n={len(sec)}) ===")
print(f"{'feature set':16}{'Linear':>9}{'RandForest':>12}{'GradBoost':>11}")
for s,cols in SETS.items():
    print(f"{s:16}" + "".join(f"{lodo(m,cols):>{w}.3f}" for m,w in zip(MODELS.values(),(9,12,11))))
# feature importance of the full set (RF)
rf=RandomForestRegressor(n_estimators=600,min_samples_leaf=3,random_state=0,n_jobs=-1).fit(
    sec[ENV+TERR].fillna(sec[ENV+TERR].mean()).values, y)
imp=pd.Series(rf.feature_importances_, index=ENV+TERR).sort_values(ascending=False)
print("\nRF feature importance (ENV+terrain, in-sample fit):")
print(imp.round(3).to_string())
