"""Gridded version of the within-district model (temp + habitat + built-up).

Habitat (rice+wetland) and built-up are recomputed as FOCAL fractions on a 100 m
grid (moving-window % around each pixel) so they have meaning at pixel scale.
Because the model is linear, fitting the coefficients on the POP-WEIGHTED SECTOR
MEANS of the pixel features makes the gridded map exactly consistent: the
pop-weighted mean over a sector of (beta . pixel_features) equals the sector
prediction. Validate within-district concordance vs temperature (paired
bootstrap); pick the focal window; render the pixel risk surface.
"""
import numpy as np, pandas as pd, xarray as xr
import rasterio.features as rfeat
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import build_grid, reproject_to, reproject_population_to
from chap_gis.io import chelsa

RES = 100.0
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
aoi = cgis.aoi.buffered(gdf, 0.0027)
grid = build_grid(gdf, resolution=RES / 111_000, crs="EPSG:4326")
gshape = grid.rio.shape if hasattr(grid.rio, "shape") else grid.shape

print("loading + reprojecting rasters to 100 m grid ...", flush=True)
land = chunk(cgis.io.worldcover.load(aoi=aoi, start=2021, end=2021, country_code="RWA"))
lc = np.asarray(reproject_to(land, grid, "mode").compute().values)
if lc.ndim == 3: lc = lc[0]
rice = cgis.io.rice.load(country_code="RWA")
if "band" in rice.dims: rice = rice.squeeze("band", drop=True)
rice_g = np.asarray(reproject_to(rice.astype("float32"), grid, "average").compute().values, np.float32)
if rice_g.ndim == 3: rice_g = rice_g[0]

gdf0 = cgis.io.boundaries.load("RWA", level=0)
tas = chelsa.load(gdf0, start="2021-01", end="2021-12", country_code="RWA").mean("time")
tas = tas.rio.write_crs("EPSG:4326")
temp = np.asarray(reproject_to(tas, grid, "bilinear").compute().values, np.float32)
if temp.ndim == 3: temp = temp[0]

wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True)
wp.rio.write_crs("EPSG:4326", inplace=True)
pop = np.asarray(reproject_population_to(wp, grid, "bilinear").compute().values, np.float32)
if pop.ndim == 3: pop = pop[0]
pop = np.clip(np.nan_to_num(pop, nan=0.0), 0, None)

shp = temp.shape
hab_pres = (((rice_g > 0) | np.isin(lc, [90, 95]))).astype(np.float32)   # rice OR herbaceous wetland/mangrove
built_pres = (lc == 50).astype(np.float32)                               # built-up

sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)),
        out_shape=shp, transform=grid.rio.transform(), fill=-1, dtype="int32")
ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64)
psum = np.bincount(s, weights=w, minlength=NS)
def sec_mean(arr): return np.bincount(s, weights=w * arr[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
temp_s = sec_mean(temp)

# target + district
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and psum[i] > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]])
grp = np.array([par[loc[i]] for i in range(NS) if keep[i]])
print(f"sectors={keep.sum()} districts={pd.Series(grp).nunique()}", flush=True)

logo = LeaveOneGroupOut()
def oof_within(X):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, grp):
        gt = grp[tr]; Xt = X[tr].copy(); yt = y[tr].copy()
        for d in np.unique(gt):
            m = gt == d; Xt[m] -= Xt[m].mean(0); yt[m] -= yt[m].mean()
        pred[te] = clone(LinearRegression()).fit(Xt, yt).predict(X[te] - X[te].mean(0))
    return pred
PD = [(d, i[a], i[b]) for d in np.unique(grp) for i in [np.where(grp == d)[0]]
      for a in range(len(i)) for b in range(a + 1, len(i)) if y[i[a]] != y[i[b]]]
by_d = {d: [(a, b) for (dd, a, b) in PD if dd == d] for d in np.unique(grp)}
ALL = [(a, b) for (_, a, b) in PD]
def conc(p, pairs):
    C = Dd = 0.0
    for a, b in pairs:
        dp = p[a] - p[b]; dy = y[a] - y[b]
        if dp == 0: C += .5; Dd += .5
        elif np.sign(dp) == np.sign(dy): C += 1
        else: Dd += 1
    return C / (C + Dd)
base_c = conc(temp_s[keep], ALL)   # temperature baseline (monotone -> rank = temp rank)
print(f"\ntemperature baseline within-district concordance = {base_c:.3f}")

rng = np.random.RandomState(0); dists = np.unique(grp)
def paired_P(p):
    diffs = []
    for _ in range(1000):
        ds = rng.choice(dists, len(dists), True); pr = [pp for d in ds for pp in by_d[d]]
        diffs.append(conc(p, pr) - conc(temp_s[keep], pr))
    d = np.array(diffs); return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()

print(f"\n{'focal window':14}{'concord':>9}{'gain':>8}{'P(>temp)':>10}")
best = None
for km in [0.5, 1.0, 2.0]:
    W = int(round(km * 1000 / RES)) | 1                        # odd window in pixels
    # transform per pixel FIRST, then aggregate -> pop-weighted mean of the pixel
    # feature == sector feature, so the linear pixel map re-aggregates exactly.
    fhL = np.log1p(ndimage.uniform_filter(hab_pres, size=W, mode="nearest"))
    fbL = np.log1p(ndimage.uniform_filter(built_pres, size=W, mode="nearest"))
    hab_s = sec_mean(fhL); built_s = sec_mean(fbL)
    X = np.column_stack([temp_s, hab_s, built_s])[keep]
    p = oof_within(X); c = conc(p, ALL); m, ci, P = paired_P(p)
    print(f"{str(km)+' km':14}{c:>9.3f}{c-base_c:>+8.3f}{P:>10.3f}   CI[{ci[0]:+.3f},{ci[1]:+.3f}]")
    if best is None or c > best["c"]:
        best = dict(km=km, W=W, fhL=fhL, fbL=fbL, hab_s=hab_s, built_s=built_s, c=c, P=P)

print(f"\nbest focal window = {best['km']} km : concordance {best['c']:.3f} vs temp {base_c:.3f}  P={best['P']:.3f}", flush=True)

# fit coefficients on all data (within-estimator) for the chosen window -> pixel risk surface
Xb = np.column_stack([temp_s, best["hab_s"], best["built_s"]])[keep]
Xd = Xb.copy(); yd = y.copy()
for d in np.unique(grp):
    m = grp == d; Xd[m] -= Xd[m].mean(0); yd[m] -= yd[m].mean()
beta = LinearRegression(fit_intercept=False).fit(Xd, yd).coef_
print(f"within-estimator coefficients [temp, log_habitat, log_built] = {np.round(beta,4)}")

risk_pix = np.full(shp, np.nan, np.float32)
risk_pix[ok] = (beta[0] * temp[ok] + beta[1] * best["fhL"][ok] + beta[2] * best["fbL"][ok])

# verify cross-scale consistency: pop-weighted sector mean of pixel risk == beta . sector features
sec_from_pix = sec_mean(np.where(np.isfinite(risk_pix), risk_pix, 0.0))
sec_from_model = Xb @ beta
maxerr = np.max(np.abs(sec_from_pix[keep] - sec_from_model))
print(f"consistency check: max |sector mean of grid  -  model prediction| = {maxerr:.2e}")
ras = xr.DataArray(risk_pix, dims=("y", "x"),
                   coords={"y": grid["y"].values, "x": grid["x"].values}, name="within_district_risk")
ras = ras.rio.write_crs("EPSG:4326"); ras.rio.write_nodata(np.nan, inplace=True)
ras.to_netcdf("results/gridded_within_risk.nc"); ras.rio.to_raster("results/gridded_within_risk.tif")

fig, ax = plt.subplots(figsize=(9, 8))
vlo, vhi = np.nanpercentile(risk_pix, [2, 98])
im = ax.imshow(risk_pix, cmap="RdYlGn_r", vmin=vlo, vmax=vhi)
ax.set_title(f"Gridded within-district risk (100 m, focal {best['km']} km)\n"
             f"sector-validated concordance {best['c']:.3f} vs temperature {base_c:.3f}", fontsize=11)
ax.axis("off"); fig.colorbar(im, ax=ax, shrink=.7, label="within-district risk index")
fig.tight_layout(); fig.savefig("results/gridded_within_risk.png", dpi=150, bbox_inches="tight")
print("wrote results/gridded_within_risk.{nc,tif,png}", flush=True)
