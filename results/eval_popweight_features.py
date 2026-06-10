"""Does POPULATION-WEIGHTING the raster aggregation improve sector-level skill?

Controlled A/B: aggregate the same temp+vegetation rasters (stack.nc: CHELSA
temperature + MODIS NDVI/EVI, 2021 annual means) to sectors two ways
  - AREA: simple unweighted pixel mean over the sector polygon
  - POP : population-weighted pixel mean (WorldPop 2021)
then compare leave-one-district-out Spearman vs incidence for each weighting,
per feature family (temp-only, veg-only, temp+veg) and model (Linear / RF).

The mechanistic hypothesis: disease arises where people live, so a pop-weighted
mean reflects the conditions humans actually experience, whereas an area mean is
diluted by uninhabited terrain (steep highlands, parks). If true, POP > AREA.
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

# ---- rasters (annual means) ----
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

ok = (sect >= 0) & np.isfinite(T) & np.isfinite(Nd) & np.isfinite(Ev) & np.isfinite(pop) & (Nd > -1)
s = sect[ok]
w = pop[ok].astype(np.float64)
ones = np.ones_like(w)
acnt = np.bincount(s, weights=ones, minlength=NS)        # pixel count (area weight)
psum = np.bincount(s, weights=w, minlength=NS)           # population per sector

def agg(arr, weights, norm):
    return np.bincount(s, weights=weights * arr[ok].astype(np.float64), minlength=NS) / np.maximum(norm, 1e-9)

feat = {}
for name, arr in (("temp", T), ("ndvi", Nd), ("evi", Ev)):
    feat[f"{name}_area"] = agg(arr, ones, acnt)
    feat[f"{name}_pop"]  = agg(arr, w, psum)
df = pd.DataFrame(feat, index=loc); df.index.name = "location_id"

# ---- target + district groups ----
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
df = df.join(tgt).join(par.rename("parent"))
df = df[df["annual_incidence_per1000"].notna() & df["parent"].notna() & (psum[[list(loc).index(x) for x in df.index]] > 0)]
y = df["annual_incidence_per1000"].values; groups = df["parent"].astype(str).values
print(f"sectors={len(df)}  districts={pd.Series(groups).nunique()}", flush=True)

# ---- how different are the two weightings? ----
print("\n=== area-mean vs pop-mean of each raster ===")
print(f"{'feature':6}{'Spearman(area,pop)':>20}{'mean(pop-area)':>16}")
for name in ("temp", "ndvi", "evi"):
    a, p = df[f"{name}_area"], df[f"{name}_pop"]
    print(f"{name:6}{spearmanr(a, p).statistic:>20.3f}{(p-a).mean():>16.3f}")

# ---- LODO skill: AREA vs POP, per family x model ----
logo = LeaveOneGroupOut()
def lodo(model, cols):
    X = df[cols].fillna(df[cols].mean()).values; pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(pred, y).statistic

MODELS = {"Linear": LinearRegression(),
          "RandomForest": RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=0, n_jobs=-1)}
FAMILIES = {
    "temp":     ["temp"],
    "veg":      ["ndvi", "evi"],
    "temp+veg": ["temp", "ndvi", "evi"],
}
print("\n=== leave-one-district-out Spearman vs incidence ===")
print(f"{'family':10}{'weighting':10}" + "".join(f"{m:>14}" for m in MODELS))
for fam, bases in FAMILIES.items():
    for wname in ("area", "pop"):
        cols = [f"{b}_{wname}" for b in bases]
        row = "".join(f"{lodo(m, cols):>14.3f}" for m in MODELS.values())
        print(f"{fam:10}{wname:10}{row}")
    # combined area+pop (does pop-weighting ADD signal beyond area?)
    cols = [f"{b}_{wname}" for b in bases for wname in ("area", "pop")]
    row = "".join(f"{lodo(m, cols):>14.3f}" for m in MODELS.values())
    print(f"{fam:10}{'both':10}{row}")

df.to_csv("results/rwanda_sector_popweight_features.csv")
print("\nwrote results/rwanda_sector_popweight_features.csv", flush=True)
