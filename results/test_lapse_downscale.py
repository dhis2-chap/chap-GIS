"""Lapse-rate downscale CHELSA temperature with the 30 m DEM, test if it improves
the risk map. CHELSA (~1 km, bilinear) flattens sub-km topographic temperature;
in rugged Rwanda valley floors are warmer than ridges -> potential within-district
signal the coarse field misses.

T_ds(pixel) = T_coarse + Gamma * (z_coarse - z_fine),  Gamma = lapse rate (C/m),
z_coarse = ~1 km focal-mean elevation (what the coarse temp 'sees'), z_fine = 100 m
DEM. Gamma fitted empirically from coarse temp vs elevation. This conserves the
~1 km mean and injects topographic detail. Evaluate within-district concordance
AND burden-capture, temp-only and full model, vs the coarse temperature.
"""
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries, chunk
from chap_gis.grid import build_grid, reproject_to

RES = 100.0
z = np.load("results/_habitat_raw.npz", allow_pickle=True)
temp, pop, sect = z["temp"], z["pop"], z["sect"]; rice_p, wet_p, built_p = z["rice_p"], z["wet_p"], z["built_p"]
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
aoi = cgis.aoi.buffered(gdf, 0.0027); grid = build_grid(gdf, resolution=RES / 111_000, crs="EPSG:4326")
elev = chunk(cgis.io.elevation.load(aoi=aoi, country_code="RWA"))
zf = np.asarray(reproject_to(elev, grid, "bilinear").compute().values, np.float32); zf = zf[0] if zf.ndim == 3 else zf
assert zf.shape == temp.shape, (zf.shape, temp.shape)

W = 11   # ~1.1 km focal -> CHELSA cell elevation
zc = ndimage.uniform_filter(np.nan_to_num(zf, nan=float(np.nanmean(zf))), size=W, mode="nearest")
fin = np.isfinite(temp) & np.isfinite(zf) & (sect >= 0)
gamma = np.polyfit(zc[fin], temp[fin], 1)[0]            # C per m (negative)
temp_ds = temp + gamma * (zf - zc)                       # warmer in valleys, cooler on ridges
print(f"fitted lapse rate = {gamma*1000:.2f} C/km", flush=True)

ok = fin & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64); psum = np.bincount(s, weights=w, minlength=NS)
def sm(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
# within-sector temp spread added by downscaling
addstd = np.sqrt(np.maximum(sm(temp_ds**2) - sm(temp_ds)**2, 0)) - np.sqrt(np.maximum(sm(temp**2) - sm(temp)**2, 0))
print(f"mean extra within-sector temp std from downscaling = {np.nanmean(addstd):.3f} C", flush=True)
hab_s, blt_s = sm(np.log1p(ndimage.uniform_filter(np.maximum(rice_p, wet_p), 5, mode="nearest"))), \
               sm(np.log1p(ndimage.uniform_filter(built_p.astype(np.float32), 5, mode="nearest")))
tc, td = sm(temp), sm(temp_ds)
sgc, sgd = sm(expit((temp - 19) / .5)), sm(expit((temp_ds - 19) / .5))

# health + target + parent
sw = pd.read_csv("results/rwanda_sweep_temp.csv"); sw["location_id"] = sw.location_id.astype(str)
hd = sw.groupby("location_id").agg(cases=("disease", "sum"), pp=("population", "mean"))
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
D = pd.DataFrame({"location_id": loc, "tc": tc, "td": td, "sgc": sgc, "sgd": sgd, "hab": hab_s, "blt": blt_s, "psum": psum})
D = D.merge(hd, left_on="location_id", right_index=True).merge(tgt.rename("inc"), left_on="location_id", right_index=True)
D = D.merge(par.rename("parent"), left_on="location_id", right_index=True)
D = D[(D.psum > 0) & D.parent.notna() & D.inc.notna()].reset_index(drop=True)
y = D["inc"].values; grp = D["parent"].astype(str).values; pop_s = D["pp"].values; cas = D["cases"].values
logo = LeaveOneGroupOut()
def oof_within(cols):
    X = D[cols].values; p = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, grp):
        gt = grp[tr]; Xt = X[tr].copy(); yt = y[tr].copy()
        for d in np.unique(gt):
            m = gt == d; Xt[m] -= Xt[m].mean(0); yt[m] -= yt[m].mean()
        p[te] = clone(LinearRegression()).fit(Xt, yt).predict(X[te] - X[te].mean(0))
    return p
def oof_pool(cols):
    X = D[cols].values; p = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, grp): p[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    return p
PD = [(d, i[a], i[b]) for d in np.unique(grp) for i in [np.where(grp == d)[0]] for a in range(len(i)) for b in range(a+1, len(i)) if y[i[a]] != y[i[b]]]
byd = {d: [(a, b) for (dd, a, b) in PD if dd == d] for d in np.unique(grp)}; ALL = [(a, b) for (_, a, b) in PD]
def conc(p, pairs):
    C = Dd = 0.0
    for a, b in pairs:
        dp = p[a]-p[b]; dy = y[a]-y[b]
        if dp == 0: C += .5; Dd += .5
        elif np.sign(dp) == np.sign(dy): C += 1
        else: Dd += 1
    return C/(C+Dd)
def cap(sc, frac, P=pop_s, Cc=cas):
    o = np.argsort(-sc); cp = np.concatenate([[0], np.cumsum(P[o])/P.sum()]); cc = np.concatenate([[0], np.cumsum(Cc[o])/Cc.sum()])
    return np.interp(frac, cp, cc)
rng = np.random.RandomState(0); dists = np.unique(grp); idxby = {d: np.where(grp == d)[0] for d in dists}
def pboot_conc(p, ref):
    v = []
    for _ in range(1500):
        ds = rng.choice(dists, len(dists), True); pr = [pp for d in ds for pp in byd[d]]
        v.append(conc(p, pr) - conc(ref, pr))
    return (np.array(v) > 0).mean()
def pboot_cap(col, ref, frac=.5):
    v = []
    for _ in range(1500):
        ds = rng.choice(dists, len(dists), True); idx = np.concatenate([idxby[d] for d in ds]); P = pop_s[idx]; Cc = cas[idx]
        v.append(cap(D[col].values[idx], frac, P, Cc) - cap(D[ref].values[idx], frac, P, Cc))
    return np.array(v)

print("\n=== WITHIN-DISTRICT CONCORDANCE ===")
ct = conc(D.tc.values, ALL); cd = conc(D.td.values, ALL)
print(f"  temp coarse (single)      {ct:.3f}")
print(f"  temp lapse-ds (single)    {cd:.3f}   P(ds>coarse)={pboot_conc(D.td.values, D.tc.values):.3f}")
fc = conc(oof_within(['tc','hab','blt']), ALL); fd = conc(oof_within(['td','hab','blt']), ALL)
print(f"  full coarse [tc,hab,blt]  {fc:.3f}")
print(f"  full lapse  [td,hab,blt]  {fd:.3f}   P(ds>coarse)={pboot_conc(oof_within(['td','hab','blt']), oof_within(['tc','hab','blt'])):.3f}")

print("\n=== BURDEN CAPTURE @50% pop ===")
print(f"  temp coarse (single)      {cap(D.tc.values,.5):.3f}")
b = pboot_cap('td','tc'); print(f"  temp lapse-ds (single)    {cap(D.td.values,.5):.3f}   gap {b.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}] P={ (b>0).mean():.3f}")
D['fc'] = oof_pool(['sgc','hab','blt']); D['fd'] = oof_pool(['sgd','hab','blt'])
print(f"  full coarse map           {cap(D.fc.values,.5):.3f}")
b2 = pboot_cap('fd','fc'); print(f"  full lapse-ds map         {cap(D.fd.values,.5):.3f}   gap {b2.mean():+.3f} [{np.percentile(b2,2.5):+.3f},{np.percentile(b2,97.5):+.3f}] P={ (b2>0).mean():.3f}")
