"""Population-by-temperature-band covariates: does the SHAPE of where people sit
on the temperature gradient add skill beyond a single mean temperature?

Per sector, fraction of population living in each annual-mean-temperature band
(CHELSA temp x WorldPop 2021). This is the direct analogue of the elevation-band
experiment, on the actual driver. Malaria turns on around ~22-23 C (report's
logistic curve), so the warm-band population share should be a sharp, interpretable
risk covariate that a single pop-weighted mean temperature blurs (a sector whose
mean is 20 C but with a hot 23 C valley where people cluster is riskier than its
mean implies).

Compare leave-one-district-out Spearman vs incidence for:
  - mean temperature only (pop-weighted & area)
  - temperature-band population fractions (the distribution)
  - +vegetation, vs the temp+veg baseline
"""
import numpy as np, xarray as xr, pandas as pd
import rasterio.features as rfeat
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries
from chap_gis.grid import reproject_population_to

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

wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True)
wp.rio.write_crs("EPSG:4326", inplace=True)
pop = reproject_population_to(wp, tann, "bilinear").values.astype(np.float32)
if pop.ndim == 3: pop = pop[0]
pop = np.clip(pop, 0, None)

ok = (sect >= 0) & np.isfinite(T) & np.isfinite(Nd) & np.isfinite(Ev) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64); t = T[ok].astype(np.float64)
psum = np.bincount(s, weights=w, minlength=NS)
acnt = np.bincount(s, weights=np.ones_like(w), minlength=NS)

tq = np.percentile(t, [5, 25, 50, 75, 95])
print(f"temperature (pixel) p5/25/50/75/95 = {np.round(tq,1)} C", flush=True)

# ---- temperature bands: fraction of POPULATION in each band, per sector ----
EDGES = [-99, 16, 18, 20, 22, 24, 99]
LABELS = [f"ptemp_{EDGES[i]}_{EDGES[i+1]}" for i in range(len(EDGES) - 1)]
LABELS[0] = "ptemp_lt16"; LABELS[-1] = "ptemp_ge24"
band = np.clip(np.digitize(t, EDGES) - 1, 0, len(EDGES) - 2)
NB = len(LABELS)
popband = np.bincount(s * NB + band, weights=w, minlength=NS * NB).reshape(NS, NB)
pfrac = popband / np.maximum(psum[:, None], 1e-9)

def agg(arr, weights, norm):
    return np.bincount(s, weights=weights * arr, minlength=NS) / np.maximum(norm, 1e-9)
temp_pop = agg(t, w, psum)
temp_area = agg(t, np.ones_like(w), acnt)

df = pd.DataFrame(pfrac, columns=LABELS, index=loc); df.index.name = "location_id"
df["temp_pop_mean"] = temp_pop
df["temp_area_mean"] = temp_area
df["pop_above_22"] = pfrac[:, 4] + pfrac[:, 5]      # headline: warm (>22 C) population share
for name, arr in (("ndvi", Nd), ("evi", Ev)):
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

kept = [list(loc).index(x) for x in df.index]
print("\n=== national population share by temperature band ===")
shares = popband[kept].sum(0) / popband[kept].sum()
for lab, sh in zip(LABELS, shares):
    print(f"  {lab:14}{sh*100:6.1f}%")

print("\n=== single-feature spatial Spearman vs incidence ===")
for c in ["pop_above_22", "temp_pop_mean", "temp_area_mean"] + LABELS:
    print(f"  {c:14}{spearmanr(df[c], y).statistic:+.3f}")

# ---- LODO skill ----
logo = LeaveOneGroupOut()
def lodo(model, cols):
    X = df[cols].fillna(df[cols].mean()).values; pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(pred, y).statistic

MODELS = {"Linear": LinearRegression(),
          "RandomForest": RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=0, n_jobs=-1)}
VEG = ["ndvi_pop", "evi_pop"]
SETS = {
    "temp mean (pop)":         ["temp_pop_mean"],
    "pop_above_22":            ["pop_above_22"],
    "temp bands (dist.)":      LABELS,
    "temp mean + bands":       ["temp_pop_mean"] + LABELS,
    "temp mean + veg":         ["temp_pop_mean"] + VEG,
    "temp bands + veg":        LABELS + VEG,
}
print("\n=== leave-one-district-out Spearman vs incidence ===")
print(f"{'feature set':22}" + "".join(f"{m:>14}" for m in MODELS))
for n, cols in SETS.items():
    print(f"{n:22}" + "".join(f"{lodo(m, cols):>14.3f}" for m in MODELS.values()))

df.to_csv("results/rwanda_sector_temp_pop_bins.csv")
print("\nwrote results/rwanda_sector_temp_pop_bins.csv", flush=True)
