"""Sigmoid temperature transform for the gridded within-district risk map.

Replace the linear temperature term with sigmoid((T-T0)/k), applied PER PIXEL
then pop-weight-aggregated to sectors (keeps cross-scale consistency). Sweep
(T0, k); compare within-district concordance to the linear-temperature gridded
model and the raw-temperature baseline; paired district-bootstrap for
reliability; render the improved grid if a sigmoid reliably wins.
"""
import numpy as np, pandas as pd, xarray as xr
from pathlib import Path
import rasterio.features as rfeat
from scipy import ndimage
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import build_grid, reproject_to, reproject_population_to
from chap_gis.io import chelsa

RES = 100.0; KM = 0.5
CACHE = Path("results/_gridded_arrays.npz")
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)

if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    temp, fhL, fbL, pop, sect = z["temp"], z["fhL"], z["fbL"], z["pop"], z["sect"]
    gx, gy = z["gx"], z["gy"]
    print("loaded cached grid arrays", flush=True)
else:
    aoi = cgis.aoi.buffered(gdf, 0.0027)
    grid = build_grid(gdf, resolution=RES / 111_000, crs="EPSG:4326")
    print("loading + reprojecting rasters ...", flush=True)
    land = chunk(cgis.io.worldcover.load(aoi=aoi, start=2021, end=2021, country_code="RWA"))
    lc = np.asarray(reproject_to(land, grid, "mode").compute().values)
    if lc.ndim == 3: lc = lc[0]
    rice = cgis.io.rice.load(country_code="RWA")
    if "band" in rice.dims: rice = rice.squeeze("band", drop=True)
    rice_g = np.asarray(reproject_to(rice.astype("float32"), grid, "average").compute().values, np.float32)
    if rice_g.ndim == 3: rice_g = rice_g[0]
    gdf0 = cgis.io.boundaries.load("RWA", level=0)
    tas = chelsa.load(gdf0, start="2021-01", end="2021-12", country_code="RWA").mean("time").rio.write_crs("EPSG:4326")
    temp = np.asarray(reproject_to(tas, grid, "bilinear").compute().values, np.float32)
    if temp.ndim == 3: temp = temp[0]
    wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True); wp.rio.write_crs("EPSG:4326", inplace=True)
    pop = np.asarray(reproject_population_to(wp, grid, "bilinear").compute().values, np.float32)
    if pop.ndim == 3: pop = pop[0]
    pop = np.clip(np.nan_to_num(pop, nan=0.0), 0, None)
    W = int(round(KM * 1000 / RES)) | 1
    fhL = np.log1p(ndimage.uniform_filter((((rice_g > 0) | np.isin(lc, [90, 95]))).astype(np.float32), size=W, mode="nearest"))
    fbL = np.log1p(ndimage.uniform_filter((lc == 50).astype(np.float32), size=W, mode="nearest"))
    sect = rfeat.rasterize(((g, i) for i, g in enumerate(gdf.geometry)), out_shape=temp.shape,
            transform=grid.rio.transform(), fill=-1, dtype="int32")
    gx, gy = grid["x"].values, grid["y"].values
    np.savez(CACHE, temp=temp, fhL=fhL, fbL=fbL, pop=pop, sect=sect, gx=gx, gy=gy)
    print("cached grid arrays", flush=True)

shp = temp.shape
ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64)
psum = np.bincount(s, weights=w, minlength=NS)
def sec_mean(arr): return np.bincount(s, weights=w * arr[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
temp_s, hab_s, built_s = sec_mean(temp), sec_mean(fhL), sec_mean(fbL)

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and psum[i] > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]])
grp = np.array([par[loc[i]] for i in range(NS) if keep[i]])
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

base_temp = conc(temp_s[keep], ALL)                                          # raw temperature
lin = oof_within(np.column_stack([temp_s, hab_s, built_s])[keep]); c_lin = conc(lin, ALL)   # current gridded (linear temp)
print(f"raw-temperature baseline concordance      = {base_temp:.3f}")
print(f"linear-temp gridded model concordance     = {c_lin:.3f}\n")

# sweep sigmoid (T0, k); aggregate sigmoid PER PIXEL
print(f"{'sigmoid temp':22}{'concord':>9}{'vs linear':>11}")
results = []
for T0 in [18, 19, 20, 21, 22, 23, 24]:
    for k in [0.5, 1.0, 2.0, 3.0]:
        sig_s = sec_mean(expit((temp - T0) / k))
        p = oof_within(np.column_stack([sig_s, hab_s, built_s])[keep]); c = conc(p, ALL)
        results.append((T0, k, c, p))
results.sort(key=lambda r: -r[2])
for T0, k, c, _ in results[:8]:
    print(f"  T0={T0} k={k:<4}        {c:>9.3f}{c-c_lin:>+11.3f}")
bestT0, bestk, bestc, bestp = results[0]

rng = np.random.RandomState(0); dists = np.unique(grp)
def paired_P(p, ref):
    diffs = []
    for _ in range(1500):
        ds = rng.choice(dists, len(dists), True); pr = [pp for d in ds for pp in by_d[d]]
        diffs.append(conc(p, pr) - conc(ref, pr))
    d = np.array(diffs); return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()
m1, ci1, P1 = paired_P(bestp, lin)              # vs linear-temp gridded model
m2, ci2, P2 = paired_P(bestp, temp_s[keep])     # vs raw temperature
print(f"\nbest sigmoid: T0={bestT0}, k={bestk}  concordance={bestc:.3f}")
print(f"  vs LINEAR-temp gridded: gain {m1:+.3f}  CI[{ci1[0]:+.3f},{ci1[1]:+.3f}]  P(>linear)={P1:.3f}")
print(f"  vs RAW temperature    : gain {m2:+.3f}  CI[{ci2[0]:+.3f},{ci2[1]:+.3f}]  P(>temp)  ={P2:.3f}")

# fit + render the sigmoid gridded map
sig_s = sec_mean(expit((temp - bestT0) / bestk))
Xb = np.column_stack([sig_s, hab_s, built_s])[keep]; Xd = Xb.copy(); yd = y.copy()
for d in np.unique(grp):
    m = grp == d; Xd[m] -= Xd[m].mean(0); yd[m] -= yd[m].mean()
beta = LinearRegression(fit_intercept=False).fit(Xd, yd).coef_
print(f"coefficients [sigmoid_temp, log_habitat, log_built] = {np.round(beta,3)}")
risk = np.full(shp, np.nan, np.float32)
sigpix = expit((temp - bestT0) / bestk)
risk[ok] = beta[0] * sigpix[ok] + beta[1] * fhL[ok] + beta[2] * fbL[ok]
sec_pix = sec_mean(np.where(np.isfinite(risk), risk, 0.0)); print(f"consistency max err = {np.max(np.abs(sec_pix[keep]-Xb@beta)):.2e}")
ras = xr.DataArray(risk, dims=("y", "x"), coords={"y": gy, "x": gx}, name="risk_sigmoid").rio.write_crs("EPSG:4326")
ras.rio.write_nodata(np.nan, inplace=True); ras.to_netcdf("results/gridded_within_risk_sigmoid.nc"); ras.rio.to_raster("results/gridded_within_risk_sigmoid.tif")
fig, ax = plt.subplots(figsize=(9, 8)); vlo, vhi = np.nanpercentile(risk, [2, 98])
im = ax.imshow(risk, cmap="RdYlGn_r", vmin=vlo, vmax=vhi)
ax.set_title(f"Gridded risk, sigmoid temp (T0={bestT0}, k={bestk})\nconcordance {bestc:.3f} vs linear {c_lin:.3f} vs temp {base_temp:.3f}", fontsize=11)
ax.axis("off"); fig.colorbar(im, ax=ax, shrink=.7, label="risk index"); fig.tight_layout()
fig.savefig("results/gridded_within_risk_sigmoid.png", dpi=150, bbox_inches="tight")
print("wrote results/gridded_within_risk_sigmoid.{nc,tif,png}", flush=True)
