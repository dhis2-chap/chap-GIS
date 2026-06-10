"""Sweep the habitat/built-up feature parameters: focal radius (separate per
feature), kernel shape (uniform vs gaussian), and habitat composition
(rice+wetland vs +permanent-water vs rice/wetland split). Score within-district
concordance (ranking, within-estimator) AND sigmoid-temp calibration R2 (pooled).
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import ndimage
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import build_grid, reproject_to, reproject_population_to
from chap_gis.io import chelsa

RES = 100.0
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
CACHE = Path("results/_habitat_raw.npz")
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    temp, pop, sect = z["temp"], z["pop"], z["sect"]
    rice_p, wet_p, water_p, built_p = z["rice_p"], z["wet_p"], z["water_p"], z["built_p"]
    print("loaded cached raw presence arrays", flush=True)
else:
    aoi = cgis.aoi.buffered(gdf, 0.0027); grid = build_grid(gdf, resolution=RES / 111_000, crs="EPSG:4326")
    print("loading rasters ...", flush=True)
    land = chunk(cgis.io.worldcover.load(aoi=aoi, start=2021, end=2021, country_code="RWA"))
    lc = np.asarray(reproject_to(land, grid, "mode").compute().values);  lc = lc[0] if lc.ndim == 3 else lc
    rice = cgis.io.rice.load(country_code="RWA")
    if "band" in rice.dims: rice = rice.squeeze("band", drop=True)
    rg = np.asarray(reproject_to(rice.astype("float32"), grid, "average").compute().values, np.float32); rg = rg[0] if rg.ndim == 3 else rg
    gdf0 = cgis.io.boundaries.load("RWA", level=0)
    tas = chelsa.load(gdf0, start="2021-01", end="2021-12", country_code="RWA").mean("time").rio.write_crs("EPSG:4326")
    temp = np.asarray(reproject_to(tas, grid, "bilinear").compute().values, np.float32); temp = temp[0] if temp.ndim == 3 else temp
    wp = cgis.io.worldpop.load(country_code="RWA", start=2021, end=2021).squeeze(drop=True); wp.rio.write_crs("EPSG:4326", inplace=True)
    pop = np.asarray(reproject_population_to(wp, grid, "bilinear").compute().values, np.float32); pop = pop[0] if pop.ndim == 3 else pop
    pop = np.clip(np.nan_to_num(pop, nan=0.0), 0, None)
    rice_p = (rg > 0).astype(np.float32); wet_p = np.isin(lc, [90, 95]).astype(np.float32)
    water_p = (lc == 80).astype(np.float32); built_p = (lc == 50).astype(np.float32)
    sect = np.ascontiguousarray(__import__("rasterio").features.rasterize(
        ((g, i) for i, g in enumerate(gdf.geometry)), out_shape=temp.shape,
        transform=grid.rio.transform(), fill=-1, dtype="int32"))
    np.savez(CACHE, temp=temp, pop=pop, sect=sect, rice_p=rice_p, wet_p=wet_p, water_p=water_p, built_p=built_p)
    print("cached raw presence arrays", flush=True)

ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64); psum = np.bincount(s, weights=w, minlength=NS)
def sec_mean(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
temp_s = sec_mean(temp)
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and psum[i] > 0 for i in range(NS)])
keepc = keep & np.array([tgt.get(loc[i], 9e9) <= 1000 for i in range(NS)])    # calibration excludes artifacts
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]]); grp = np.array([par[loc[i]] for i in range(NS) if keep[i]])
yc = np.array([tgt[loc[i]] for i in range(NS) if keepc[i]]); grpc = np.array([par[loc[i]] for i in range(NS) if keepc[i]])
logo = LeaveOneGroupOut()

def focal(pres, km, kernel):
    if kernel == "uniform":
        return ndimage.uniform_filter(pres, size=int(round(km * 1000 / RES)) | 1, mode="nearest")
    return ndimage.gaussian_filter(pres, sigma=km * 1000 / RES / 2.0, mode="nearest")

def oof_within(X, yy, gg):
    pred = np.full(len(yy), np.nan)
    for tr, te in logo.split(X, yy, gg):
        gt = gg[tr]; Xt = X[tr].copy(); yt = yy[tr].copy()
        for d in np.unique(gt):
            m = gt == d; Xt[m] -= Xt[m].mean(0); yt[m] -= yt[m].mean()
        pred[te] = clone(LinearRegression()).fit(Xt, yt).predict(X[te] - X[te].mean(0))
    return pred
PD = [(d, i[a], i[b]) for d in np.unique(grp) for i in [np.where(grp == d)[0]]
      for a in range(len(i)) for b in range(a + 1, len(i)) if y[i[a]] != y[i[b]]]
ALL = [(a, b) for (_, a, b) in PD]
def conc(p):
    C = Dd = 0.0
    for a, b in ALL:
        dp = p[a] - p[b]; dy = y[a] - y[b]
        if dp == 0: C += .5; Dd += .5
        elif np.sign(dp) == np.sign(dy): C += 1
        else: Dd += 1
    return C / (C + Dd)
def oof_pool(X, yy, gg):
    pred = np.full(len(yy), np.nan)
    for tr, te in logo.split(X, yy, gg):
        pred[te] = clone(LinearRegression()).fit(X[tr], yy[tr]).predict(X[te])
    return pred
def r2c(p): return 1 - np.sum((yc - p) ** 2) / np.sum((yc - yc.mean()) ** 2)
sigc = sec_mean(expit((temp - 19.0) / 0.5))     # fixed best temp transform for calibration

def feats(hab_pres_list, hab_r, built_r, kernel):
    fhs = [np.log1p(focal(hp, hab_r, kernel)) for hp in hab_pres_list]
    fb = np.log1p(focal(built_p, built_r, kernel))
    hab_cols = [sec_mean(fh) for fh in fhs]; bs = sec_mean(fb)
    return hab_cols, bs

print(f"\n{'config':46}{'concord':>9}{'calibR2':>9}")
def score(name, hab_list, hab_r, built_r, kernel):
    hab_cols, bs = feats(hab_list, hab_r, built_r, kernel)
    Xc = np.column_stack([temp_s] + hab_cols + [bs])
    cc = conc(oof_within(Xc[keep], y, grp))
    Xr = np.column_stack([sigc] + hab_cols + [bs])
    rr = r2c(oof_pool(Xr[keepc], yc, grpc))
    print(f"{name:46}{cc:>9.3f}{rr:>9.3f}")
    return cc, rr

# baseline (current): rice|wet combined, 0.5 km uniform, built 0.5 km
score("BASELINE rice|wet 0.5km, built 0.5km, uniform", [np.maximum(rice_p, wet_p)], 0.5, 0.5, "uniform")
print("-- habitat radius (built fixed 0.5, uniform, rice|wet) --")
for hr in [0.25, 1.0, 2.0]:
    score(f"  hab {hr}km", [np.maximum(rice_p, wet_p)], hr, 0.5, "uniform")
print("-- built radius (hab fixed 0.5) --")
for br in [0.25, 1.0]:
    score(f"  built {br}km", [np.maximum(rice_p, wet_p)], 0.5, br, "uniform")
print("-- composition --")
score("  +permanent water", [np.maximum.reduce([rice_p, wet_p, water_p])], 0.5, 0.5, "uniform")
score("  rice / wetland split (2 features)", [rice_p, wet_p], 0.5, 0.5, "uniform")
score("  rice / wetland / water split (3)", [rice_p, wet_p, water_p], 0.5, 0.5, "uniform")
print("-- kernel --")
score("  gaussian kernel 0.5km", [np.maximum(rice_p, wet_p)], 0.5, 0.5, "gaussian")
score("  gaussian kernel 1km", [np.maximum(rice_p, wet_p)], 1.0, 1.0, "gaussian")
print("-- combined best-guess: split habitat 1km + built 0.25km --")
score("  split rice/wet 1km, built 0.25km", [rice_p, wet_p], 1.0, 0.25, "uniform")
print("\nref baseline: concordance 0.706, calibration R2 0.402 (sigmoid temp T0=19,k=0.5)")
