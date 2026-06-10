"""Population-by-elevation-band covariates: does the SHAPE of where people sit
on the elevation gradient add skill beyond a single mean?

Per sector, compute the fraction of population living in each elevation band
(WorldPop 2021 x SRTM/elevation). Malaria is a low/warm-elevation disease, so
the low-band population share should be a direct, interpretable risk covariate
that a single pop-weighted mean elevation cannot fully capture (two sectors with
the same mean can have very different low-elevation tails).

Compare leave-one-district-out Spearman vs incidence for:
  - mean elevation only (pop-weighted & area)
  - elevation-band population fractions (the distribution)
  - temp+veg baseline, and temp+veg + elevation bands
"""
import numpy as np, xarray as xr, pandas as pd
import rasterio.features as rfeat
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import reproject_to, reproject_population_to

# ---- rasters on the MODIS/CHELSA grid (co-registered with pop) ----
ds = xr.open_dataset("results/veg_temp_2021/stack.nc")
tann = ds["temperature"].mean("month").rio.write_crs("EPSG:4326")
nann = ds["ndvi"].mean("month").rio.write_crs("EPSG:4326")
eann = ds["evi"].mean("month").rio.write_crs("EPSG:4326")
T, Nd, Ev = (a.values.astype(np.float32) for a in (tann, nann, eann))
shp = T.shape

gdf = prepare_boundaries("RWA", 5); NS = len(gdf); loc = gdf["location_id"].to_numpy()
sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=shp, transform=tann.rio.transform(), fill=-1, dtype="int32")

aoi = cgis.aoi.buffered(gdf, 0.0027)
elev = chunk(cgis.io.elevation.load(aoi=aoi, country_code="RWA"))
elev = reproject_to(elev, tann, "bilinear")
Z = np.asarray(elev.compute().values, np.float32)
if Z.ndim == 3: Z = Z[0]

wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True)
wp.rio.write_crs("EPSG:4326", inplace=True)
pop = reproject_population_to(wp, tann, "bilinear").values.astype(np.float32)
if pop.ndim == 3: pop = pop[0]
pop = np.clip(pop, 0, None)

ok = (sect >= 0) & np.isfinite(T) & np.isfinite(Nd) & np.isfinite(Ev) & np.isfinite(Z) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64); z = Z[ok].astype(np.float64)
psum = np.bincount(s, weights=w, minlength=NS)
acnt = np.bincount(s, weights=np.ones_like(w), minlength=NS)

# describe pixel elevation to motivate the bands
zq = np.percentile(z, [5, 25, 50, 75, 95])
print(f"elevation (pixel) p5/25/50/75/95 = {np.round(zq).astype(int)} m", flush=True)

# ---- elevation bands: fraction of POPULATION in each band, per sector ----
EDGES = [0, 1400, 1600, 1800, 2000, 2200, 9000]
LABELS = [f"pelev_{EDGES[i]}_{EDGES[i+1]}" for i in range(len(EDGES) - 1)]
band = np.clip(np.digitize(z, EDGES) - 1, 0, len(EDGES) - 2)
NB = len(LABELS)
popband = np.bincount(s * NB + band, weights=w, minlength=NS * NB).reshape(NS, NB)
pfrac = popband / np.maximum(psum[:, None], 1e-9)        # population share per band

def agg(arr, weights, norm):
    return np.bincount(s, weights=weights * arr, minlength=NS) / np.maximum(norm, 1e-9)
elev_pop = agg(z, w, psum)          # pop-weighted mean elevation
elev_area = agg(z, np.ones_like(w), acnt)

df = pd.DataFrame(pfrac, columns=LABELS, index=loc); df.index.name = "location_id"
df["elev_pop_mean"] = elev_pop
df["elev_area_mean"] = elev_area
df["pop_below_1600"] = pfrac[:, 0] + pfrac[:, 1]    # headline: low/warm-elevation population share
# temp+veg pop-weighted means (from the previous experiment's winner) for the combined set
for name, arr in (("temp", T), ("ndvi", Nd), ("evi", Ev)):
    df[f"{name}_pop"] = agg(arr[ok].astype(np.float64), w, psum)

# ---- target + district groups ----
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
df = df.join(tgt).join(par.rename("parent"))
df = df[df["annual_incidence_per1000"].notna() & df["parent"].notna()
        & (psum[[list(loc).index(x) for x in df.index]] > 0)]
y = df["annual_incidence_per1000"].values; groups = df["parent"].astype(str).values
print(f"sectors={len(df)}  districts={pd.Series(groups).nunique()}", flush=True)

print("\n=== national population share by elevation band ===")
kept = [list(loc).index(x) for x in df.index]
shares = popband[kept].sum(0) / popband[kept].sum()
for lab, sh in zip(LABELS, shares):
    print(f"  {lab:18}{sh*100:6.1f}%")

print("\n=== single-feature spatial Spearman vs incidence ===")
for c in ["pop_below_1600", "elev_pop_mean", "elev_area_mean"] + LABELS:
    print(f"  {c:18}{spearmanr(df[c], y).statistic:+.3f}")

# ---- LODO skill ----
logo = LeaveOneGroupOut()
def lodo(model, cols):
    X = df[cols].fillna(df[cols].mean()).values; pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(pred, y).statistic

MODELS = {"Linear": LinearRegression(),
          "RandomForest": RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=0, n_jobs=-1)}
TV = ["temp_pop", "ndvi_pop", "evi_pop"]
SETS = {
    "elev mean (pop)":          ["elev_pop_mean"],
    "pop_below_1600":           ["pop_below_1600"],
    "elev bands (dist.)":       LABELS,
    "elev mean + bands":        ["elev_pop_mean"] + LABELS,
    "temp+veg (baseline)":      TV,
    "temp+veg + elev mean":     TV + ["elev_pop_mean"],
    "temp+veg + elev bands":    TV + LABELS,
}
print("\n=== leave-one-district-out Spearman vs incidence ===")
print(f"{'feature set':26}" + "".join(f"{m:>14}" for m in MODELS))
for n, cols in SETS.items():
    print(f"{n:26}" + "".join(f"{lodo(m, cols):>14.3f}" for m in MODELS.values()))

df.to_csv("results/rwanda_sector_elev_pop_bins.csv")
print("\nwrote results/rwanda_sector_elev_pop_bins.csv", flush=True)
