"""Final reliably-improved gridded within-district model: temp + focal-habitat +
focal-built at 3 km (regional land-use context). Within-estimator; render the
risk surface; verify cross-scale consistency and report concordance + P(>temp).
"""
import numpy as np, pandas as pd, xarray as xr
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

RES, KM = 100.0, 3.0
z = np.load("results/_habitat_raw.npz", allow_pickle=True)
temp, pop, sect = z["temp"], z["pop"], z["sect"]; rice_p, wet_p, built_p = z["rice_p"], z["wet_p"], z["built_p"]
gz = np.load("results/_gridded_arrays.npz", allow_pickle=True); gx, gy = gz["gx"], gz["gy"]
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
W = int(round(KM * 1000 / RES)) | 1
fhL = np.log1p(ndimage.uniform_filter(np.maximum(rice_p, wet_p), size=W, mode="nearest"))
fbL = np.log1p(ndimage.uniform_filter(built_p.astype(np.float32), size=W, mode="nearest"))
ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64); psum = np.bincount(s, weights=w, minlength=NS)
def sm(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
temp_s, hab_s, blt_s = sm(temp), sm(fhL), sm(fbL)
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and psum[i] > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]]); grp = np.array([par[loc[i]] for i in range(NS) if keep[i]])
logo = LeaveOneGroupOut()
def oof(X):
    p = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, grp):
        gt = grp[tr]; Xt = X[tr].copy(); yt = y[tr].copy()
        for d in np.unique(gt):
            m = gt == d; Xt[m] -= Xt[m].mean(0); yt[m] -= yt[m].mean()
        p[te] = clone(LinearRegression()).fit(Xt, yt).predict(X[te] - X[te].mean(0))
    return p
PD = [(d, i[a], i[b]) for d in np.unique(grp) for i in [np.where(grp == d)[0]]
      for a in range(len(i)) for b in range(a + 1, len(i)) if y[i[a]] != y[i[b]]]
byd = {d: [(a, b) for (dd, a, b) in PD if dd == d] for d in np.unique(grp)}; ALL = [(a, b) for (_, a, b) in PD]
def conc(p, pairs):
    C = Dd = 0.0
    for a, b in pairs:
        dp = p[a] - p[b]; dy = y[a] - y[b]
        if dp == 0: C += .5; Dd += .5
        elif np.sign(dp) == np.sign(dy): C += 1
        else: Dd += 1
    return C / (C + Dd)
Xk = np.column_stack([temp_s, hab_s, blt_s])[keep]
base = temp_s[keep]; base_c = conc(base, ALL); p = oof(Xk); c = conc(p, ALL)
rng = np.random.RandomState(0); dists = np.unique(grp); diffs = []
for _ in range(4000):
    ds = rng.choice(dists, len(dists), True); pr = [pp for d in ds for pp in byd[d]]
    diffs.append(conc(p, pr) - conc(base, pr))
diffs = np.array(diffs); lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"temperature baseline concordance = {base_c:.3f}")
print(f"gridded (focal {KM}km) concordance = {c:.3f}   gain {c-base_c:+.3f}   95%CI[{lo:+.3f},{hi:+.3f}]   P(>temp)={(diffs>0).mean():.3f}")

# fit on all data (within-estimator) and render the pixel surface
Xd = Xk.copy(); yd = y.copy()
for d in np.unique(grp):
    m = grp == d; Xd[m] -= Xd[m].mean(0); yd[m] -= yd[m].mean()
beta = LinearRegression(fit_intercept=False).fit(Xd, yd).coef_
print(f"coefficients [temp, log_habitat, log_built] = {np.round(beta,3)}")
risk = np.full(temp.shape, np.nan, np.float32)
risk[ok] = beta[0] * temp[ok] + beta[1] * fhL[ok] + beta[2] * fbL[ok]
sec_pix = sm(np.where(np.isfinite(risk), risk, 0.0))[keep]
print(f"consistency max err = {np.max(np.abs(sec_pix - Xk @ beta)):.2e}")
ras = xr.DataArray(risk, dims=("y", "x"), coords={"y": gy, "x": gx}, name="risk").rio.write_crs("EPSG:4326")
ras.rio.write_nodata(np.nan, inplace=True)
ras.to_netcdf("results/gridded_within_risk_v2.nc"); ras.rio.to_raster("results/gridded_within_risk_v2.tif")
fig, ax = plt.subplots(figsize=(9, 8)); vlo, vhi = np.nanpercentile(risk, [2, 98])
im = ax.imshow(risk, cmap="RdYlGn_r", vmin=vlo, vmax=vhi)
ax.set_title(f"Gridded within-district risk (focal {KM}km)\nconcordance {c:.3f} vs temp {base_c:.3f}  P(>temp)={(diffs>0).mean():.3f}", fontsize=11)
ax.axis("off"); fig.colorbar(im, ax=ax, shrink=.7, label="within-district risk index"); fig.tight_layout()
fig.savefig("results/gridded_within_risk_v2.png", dpi=150, bbox_inches="tight")
print("wrote results/gridded_within_risk_v2.{nc,tif,png}", flush=True)
