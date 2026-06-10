"""Within-district mean OOF ranking skill across methods.

For each method: generate leave-one-district-out predictions, then compute the
within-district Spearman(pred, incidence) for each district (>=4 sectors) and
average over districts. This strips out the easy national highland/lowland
gradient and measures how well each method orders sectors *locally*.
Reports plain mean, Fisher-z mean, and the pooled OOF Spearman for context.
"""
import numpy as np, xarray as xr, pandas as pd
import rasterio.features as rfeat
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries
from chap_gis.grid import reproject_population_to

ds = xr.open_dataset("results/veg_temp_2021/stack.nc")
tann = ds["temperature"].mean("month").rio.write_crs("EPSG:4326")
T = tann.values.astype(np.float32)
Nd = ds["ndvi"].mean("month").values.astype(np.float32)
A = (ds["ndvi"].max("month") - ds["ndvi"].min("month")).values.astype(np.float32)
shp = T.shape
gdf = prepare_boundaries("RWA", 5); NS = len(gdf); loc = gdf["location_id"].to_numpy()
sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=shp, transform=tann.rio.transform(), fill=-1, dtype="int32")
wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True)
wp.rio.write_crs("EPSG:4326", inplace=True)
pop = reproject_population_to(wp, tann, "bilinear").values.astype(np.float32)
if pop.ndim == 3: pop = pop[0]
pop = np.clip(pop, 0, None)

ok = (sect >= 0) & np.isfinite(T) & np.isfinite(Nd) & np.isfinite(A) & np.isfinite(pop) & (Nd > -1)
s, w = sect[ok], pop[ok].astype(np.float64)
psum = np.bincount(s, weights=w, minlength=NS)
def wmean(arr): return np.bincount(s, weights=w * arr[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
M_all = np.column_stack([wmean(T), wmean(Nd), wmean(A)])

NT, NN, NA = 8, 6, 5
TB = np.linspace(10, 26, NT + 1); NBv = np.linspace(0, 0.9, NN + 1); AB = np.linspace(0, 0.6, NA + 1)
ti = np.clip(np.digitize(T, TB) - 1, 0, NT - 1); ni = np.clip(np.digitize(Nd, NBv) - 1, 0, NN - 1)
ai = np.clip(np.digitize(A, AB) - 1, 0, NA - 1)
flat = s * (NT * NN * NA) + ti[ok] * (NN * NA) + ni[ok] * NA + ai[ok]
H = np.bincount(flat, weights=w, minlength=NS * NT * NN * NA).reshape(NS, -1)
Hn_all = H / np.clip(H.sum(1, keepdims=True), 1, None)

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and H[i].sum() > 0 for i in range(NS)])
locK = loc[keep]
y = np.array([tgt[i] for i in locK]); groups = np.array([par[i] for i in locK])
M = M_all[keep]; Hn = Hn_all[keep]
H3 = Hn; H2 = Hn.reshape(len(y), NT, NN, NA).sum(3).reshape(len(y), -1); H1 = Hn.reshape(len(y), NT, NN, NA).sum((2, 3))

# full temp+veg+moisture sector feature set (the report champion)
mo = pd.read_csv("results/rwanda_sector_moisture.csv").set_index("location_id")
MOIST = ["temp_pop", "ndvi_ann", "evi_ann", "ndvi_dry", "ndvi_drop", "ndvi_ratio",
         "ndvi_amp", "evi_dry", "evi_drop", "evi_ratio", "evi_amp"]
Xmoist = mo.reindex(locK)[MOIST].values
Xmoist = np.where(np.isfinite(Xmoist), Xmoist, np.nanmean(Xmoist, axis=0))

logo = LeaveOneGroupOut()
def oof(model, X):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return pred
def concordance(pred):
    C = Dd = 0.0
    for d in np.unique(groups):
        i = np.where(groups == d)[0]
        for a in range(len(i)):
            for b in range(a + 1, len(i)):
                dy = y[i[a]] - y[i[b]]
                if dy == 0: continue
                dp = pred[i[a]] - pred[i[b]]
                if dp == 0: C += 0.5; Dd += 0.5
                elif np.sign(dp) == np.sign(dy): C += 1
                else: Dd += 1
    return C / (C + Dd) if (C + Dd) else np.nan

def skill(pred, min_n=4):
    rs = [spearmanr(pred[groups == d], y[groups == d]).statistic
          for d in np.unique(groups) if (groups == d).sum() >= min_n]
    rs = np.array([r for r in rs if np.isfinite(r)])
    fisher = np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999))))
    return concordance(pred), rs.mean(), fisher, spearmanr(pred, y).statistic, len(rs)

RF = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=0, n_jobs=-1)
GBM = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
poly2 = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression())
METHODS = [
    ("control: mean temperature",        LinearRegression(),  M[:, [0]]),
    ("linear: means(temp,ndvi,amp)",     LinearRegression(),  M),
    ("poly2: means(temp,ndvi,amp)",      poly2,               M),
    ("RF: means(temp,ndvi,amp)",         clone(RF),           M),
    ("1D surface S(temp)",               Ridge(alpha=1.0),    H1),
    ("2D surface S(temp,NDVI)",          Ridge(alpha=1.0),    H2),
    ("3D surface S(temp,NDVI,amp)",      Ridge(alpha=1.0),    H3),
    ("RF: temp+veg+moisture (11 feat)",  clone(RF),           Xmoist),
    ("GBM: temp+veg+moisture (11 feat)", clone(GBM),          Xmoist),
]
print(f"sectors={len(y)}  districts\n", flush=True)
print(f"{'method':34}{'concordance':>12}{'within-mean':>12}{'Fisher':>9}{'pooled':>9}")
res = []
for name, model, X in METHODS:
    c, m, f, p, n = skill(oof(model, X))
    res.append((name, c, m, f, p, n))
    print(f"{name:34}{c:>12.3f}{m:>12.3f}{f:>9.3f}{p:>9.3f}")
pd.DataFrame(res, columns=["method", "concordance", "within_mean", "within_fisher", "pooled", "n_districts"])\
  .sort_values("concordance", ascending=False)\
  .to_csv("results/within_district_skill_by_method.csv", index=False)
print("\nwrote results/within_district_skill_by_method.csv", flush=True)
