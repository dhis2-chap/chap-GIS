"""Refine the sigmoid (T0,k) for calibration, confirm it's not just a grid-edge
artifact, and render the calibrated gridded risk map (pooled fit, predicted
incidence per 1000). Sigmoid applied per pixel -> exact cross-scale consistency.
"""
import numpy as np, pandas as pd, xarray as xr
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

z = np.load("results/_gridded_arrays.npz", allow_pickle=True)
temp, fhL, fbL, pop, sect = z["temp"], z["fhL"], z["fbL"], z["pop"], z["sect"]; gx, gy = z["gx"], z["gy"]
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64); psum = np.bincount(s, weights=w, minlength=NS)
def sec_mean(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
temp_s, hab_s, built_s = sec_mean(temp), sec_mean(fhL), sec_mean(fbL)
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and psum[i] > 0
                 and tgt[loc[i]] <= 1000 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]]); grp = np.array([par[loc[i]] for i in range(NS) if keep[i]])
H, B = hab_s[keep], built_s[keep]
logo = LeaveOneGroupOut()
def oof(X):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, grp):
        pred[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    return pred
def r2(p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
def sigT(T0, k): return sec_mean(expit((temp - T0) / k))[keep]

print("refine grid  R2(full):")
print(f"{'k\\T0':>6}" + "".join(f"{t:>8}" for t in [16, 17, 18, 19, 20, 21]))
best = None
for k in [0.25, 0.5, 0.75, 1.0, 1.5]:
    row = []
    for T0 in [16, 17, 18, 19, 20, 21]:
        v = r2(oof(np.column_stack([sigT(T0, k), H, B])))
        row.append(v)
        if best is None or v > best[0]: best = (v, T0, k)
    print(f"{k:>6}" + "".join(f"{v:>8.3f}" for v in row))
bR2, bT0, bk = best
linR2 = r2(oof(np.column_stack([temp_s[keep], H, B])))
print(f"\nbest sigmoid: T0={bT0}, k={bk}  R2={bR2:.3f}  (linear-temp full R2={linR2:.3f})")

# pooled fit on all data -> calibrated pixel risk (predicted incidence/1000)
st = sigT(bT0, bk)
reg = LinearRegression().fit(np.column_stack([st, H, B]), y)
b0, (bt, bh, bb) = reg.intercept_, reg.coef_
shp = temp.shape; risk = np.full(shp, np.nan, np.float32)
risk[ok] = b0 + bt * expit((temp[ok] - bT0) / bk) + bh * fhL[ok] + bb * fbL[ok]
sec_from_pix = sec_mean(np.where(np.isfinite(risk), risk, 0.0))[keep]
print(f"consistency max err = {np.max(np.abs(sec_from_pix - reg.predict(np.column_stack([st,H,B])))):.2e}")
ras = xr.DataArray(risk, dims=("y", "x"), coords={"y": gy, "x": gx}, name="risk_incidence_per1000").rio.write_crs("EPSG:4326")
ras.rio.write_nodata(np.nan, inplace=True)
ras.to_netcdf("results/gridded_risk_calibrated.nc"); ras.rio.to_raster("results/gridded_risk_calibrated.tif")
fig, ax = plt.subplots(figsize=(9, 8)); vlo, vhi = np.nanpercentile(risk, [2, 98])
im = ax.imshow(np.clip(risk, 0, None), cmap="inferno", vmin=max(vlo, 0), vmax=vhi)
ax.set_title(f"Calibrated gridded risk (predicted incidence/1000)\nsigmoid temp T0={bT0},k={bk}: R2 {bR2:.3f} vs linear {linR2:.3f}", fontsize=11)
ax.axis("off"); fig.colorbar(im, ax=ax, shrink=.7, label="predicted incidence / 1000")
fig.tight_layout(); fig.savefig("results/gridded_risk_calibrated.png", dpi=150, bbox_inches="tight")
print("wrote results/gridded_risk_calibrated.{nc,tif,png}", flush=True)
