"""Does loosening the linearity help? Compare, under the same leave-one-district-out:

  A. linear on the 3 pop-weighted means (temp, NDVI, amp)          [pure linear]
  B. linear on degree-2 polynomial of the means (squares+interactions) [curvature]
  C. ridge on the 240-cell population histogram                    [current surface:
                                                                    nonparametric in
                                                                    covariates, LINEAR
                                                                    aggregation]
  D. gradient boosting on the 3 means                              [fully nonlinear, low-d]
  E. gradient boosting on the 240-cell histogram                   [nonlinear aggregation]

A/B/C isolate the two distinct linearities: B adds square terms to the *aggregate
features* (what "include square terms" usually means); C already allows arbitrary
covariate nonlinearity via bins but keeps a LINEAR (additive, inner-product)
aggregation; D/E break the aggregation linearity too.
"""
import numpy as np, xarray as xr, pandas as pd
import rasterio.features as rfeat
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
M = np.column_stack([wmean(T), wmean(Nd), wmean(A)])   # pop-weighted means: temp, ndvi, amp

NT, NN, NA = 8, 6, 5
TB = np.linspace(10, 26, NT + 1); NBv = np.linspace(0, 0.9, NN + 1); AB = np.linspace(0, 0.6, NA + 1)
ti = np.clip(np.digitize(T, TB) - 1, 0, NT - 1); ni = np.clip(np.digitize(Nd, NBv) - 1, 0, NN - 1)
ai = np.clip(np.digitize(A, AB) - 1, 0, NA - 1)
flat = s * (NT * NN * NA) + ti[ok] * (NN * NA) + ni[ok] * NA + ai[ok]
H = np.bincount(flat, weights=w, minlength=NS * NT * NN * NA).reshape(NS, -1)
Hn = H / np.clip(H.sum(1, keepdims=True), 1, None)

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and H[i].sum() > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]])
groups = np.array([par[loc[i]] for i in range(NS) if keep[i]])
Mk, Hk = M[keep], Hn[keep]
print(f"sectors={keep.sum()}  districts={pd.Series(groups).nunique()}", flush=True)

logo = LeaveOneGroupOut()
def lodo(model, X):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(pred, y).statistic

gbm = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0)
poly2 = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression())
print("\n=== LODO Spearman vs incidence ===")
rows = [
    ("A. linear on means (temp,ndvi,amp)",        LinearRegression(),                        Mk),
    ("B. linear + squares & interactions (deg2)", poly2,                                     Mk),
    ("C. ridge on histogram (current surface)",   Ridge(alpha=1.0),                          Hk),
    ("D. grad-boost on means",                    clone(gbm),                                Mk),
    ("E. grad-boost on histogram",                clone(gbm),                                Hk),
]
for name, model, X in rows:
    print(f"  {name:42}{lodo(model, X):.3f}")
print("\n  refs: linear-in-means is the most-restrictive; ridge-histogram(equal-width) was 0.758;")
print("        sector RF temp+veg+moisture ~0.748.  Spearman is rank-only (insensitive to monotone curvature).")
