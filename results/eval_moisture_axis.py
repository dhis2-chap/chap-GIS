"""Dry-season moisture-stress axis: does seasonal vegetation behaviour add a
predictive axis INDEPENDENT of mean temperature and annual-mean vegetation?

Mechanism (report sec.7): in the long dry season (Jun-Sep) warm lowland vegetation
collapses where it depends on rain, but stays green where there is permanent water
(wetlands, irrigated valleys, lakeshore) -> persistent mosquito breeding habitat.
Annual-mean NDVI and mean temperature cannot distinguish "warm + permanently moist"
from "warm + seasonally dry"; a dry-season / seasonal-amplitude feature can.

Per-pixel monthly NDVI/EVI (MODIS 2021) -> seasonal descriptors -> pop-weighted to
sectors. Evaluate (a) independence from temperature, (b) marginal correlation with
incidence, (c) whether they LIFT the temp+veg LODO baseline.
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

DRY = [6, 7, 8, 9]                 # long dry season
WET = [3, 4, 5, 10, 11, 12]        # long + short rains

ds = xr.open_dataset("results/veg_temp_2021/stack.nc")
mon = ds["month"].values
def msel(da, months): return da.sel(month=[m for m in months if m in mon])
ndvi_m = ds["ndvi"]; evi_m = ds["evi"]; temp_m = ds["temperature"]

def seasonal(da):
    ann = da.mean("month"); dry = msel(da, DRY).mean("month"); wet = msel(da, WET).mean("month")
    drop = wet - dry                                   # absolute dry-down (high = moisture-stressed)
    ratio = dry / xr.where(np.abs(wet) < 1e-3, np.nan, wet)   # persistence (high = stays green = moist)
    amp = da.max("month") - da.min("month")
    return dict(ann=ann, dry=dry, wet=wet, drop=drop, ratio=ratio, amp=amp)

NV = seasonal(ndvi_m); EV = seasonal(evi_m)
Tann = temp_m.mean("month")
tann = Tann.rio.write_crs("EPSG:4326")
shp = tann.shape

gdf = prepare_boundaries("RWA", 5); NS = len(gdf); loc = gdf["location_id"].to_numpy()
sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=shp, transform=tann.rio.transform(), fill=-1, dtype="int32")
wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True)
wp.rio.write_crs("EPSG:4326", inplace=True)
pop = reproject_population_to(wp, tann, "bilinear").values.astype(np.float32)
if pop.ndim == 3: pop = pop[0]
pop = np.clip(pop, 0, None)

T = tann.values.astype(np.float32)
ok = (sect >= 0) & np.isfinite(T) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64)
psum = np.bincount(s, weights=w, minlength=NS)
def wmean(da):
    a = np.asarray(da.values, np.float32)
    m = ok & np.isfinite(a)
    num = np.bincount(s, weights=(w * a[ok].astype(np.float64) * np.isfinite(a[ok])), minlength=NS)
    den = np.bincount(s, weights=(w * np.isfinite(a[ok])), minlength=NS)
    return num / np.maximum(den, 1e-9)

cols = {"temp_pop": wmean(tann)}
for tag, D in (("ndvi", NV), ("evi", EV)):
    for k, da in D.items():
        cols[f"{tag}_{k}"] = wmean(da)
df = pd.DataFrame(cols, index=loc); df.index.name = "location_id"

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
df = df.join(tgt).join(par.rename("parent"))
df = df[df["annual_incidence_per1000"].notna() & df["parent"].notna()
        & (psum[[list(loc).index(x) for x in df.index]] > 0)]
y = df["annual_incidence_per1000"].values; groups = df["parent"].astype(str).values
print(f"sectors={len(df)}  districts={pd.Series(groups).nunique()}", flush=True)

MOIST = ["ndvi_dry", "ndvi_drop", "ndvi_ratio", "ndvi_amp", "evi_dry", "evi_drop", "evi_ratio", "evi_amp"]
print("\n=== moisture features: independence from temp & marginal signal ===")
print(f"{'feature':12}{'rho vs temp':>13}{'rho vs incid':>14}")
for c in ["ndvi_ann", "evi_ann"] + MOIST:
    rt = spearmanr(df[c], df["temp_pop"], nan_policy="omit").statistic
    ri = spearmanr(df[c], y, nan_policy="omit").statistic
    print(f"{c:12}{rt:>13.3f}{ri:>14.3f}")

logo = LeaveOneGroupOut()
def lodo(model, cset):
    X = df[cset].fillna(df[cset].mean()).values; pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(pred, y).statistic

MODELS = {"Linear": LinearRegression(),
          "RandomForest": RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=0, n_jobs=-1)}
TV = ["temp_pop", "ndvi_ann", "evi_ann"]
SETS = {
    "temp only":              ["temp_pop"],
    "temp+veg (baseline)":    TV,
    "moisture only":          MOIST,
    "temp + moisture":        ["temp_pop"] + MOIST,
    "temp+veg + ndvi_dry":    TV + ["ndvi_dry"],
    "temp+veg + ndvi_drop":   TV + ["ndvi_drop"],
    "temp+veg + moisture":    TV + MOIST,
}
print("\n=== leave-one-district-out Spearman vs incidence ===")
print(f"{'feature set':24}" + "".join(f"{m:>14}" for m in MODELS))
for n, cset in SETS.items():
    print(f"{n:24}" + "".join(f"{lodo(m, cset):>14.3f}" for m in MODELS.values()))

rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(
    df[TV + MOIST].fillna(df[TV + MOIST].mean()).values, y)
imp = pd.Series(rf.feature_importances_, index=TV + MOIST).sort_values(ascending=False)
print("\nRF importance (temp+veg+moisture, in-sample):"); print(imp.round(3).to_string())

df.to_csv("results/rwanda_sector_moisture.csv")
print("\nwrote results/rwanda_sector_moisture.csv", flush=True)
