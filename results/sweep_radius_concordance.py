"""Sweep focal radius (habitat & built) measuring within-district concordance AND
the paired-bootstrap P(>temp). Larger radius -> features approach whole-sector
fractions (which gave P=0.96 at sector level). Find a gridded config that
reliably (P>=0.95) improves concordance over temperature.
"""
import numpy as np, pandas as pd
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

RES = 100.0
z = np.load("results/_habitat_raw.npz", allow_pickle=True)
temp, pop, sect = z["temp"], z["pop"], z["sect"]
rice_p, wet_p, built_p = z["rice_p"], z["wet_p"], z["built_p"]
gdf = prepare_boundaries("RWA", 5); loc = gdf["location_id"].to_numpy(); NS = len(gdf)
ok = (sect >= 0) & np.isfinite(temp) & np.isfinite(pop)
s = sect[ok]; w = pop[ok].astype(np.float64); psum = np.bincount(s, weights=w, minlength=NS)
def sm(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
def focal(p, km): return ndimage.uniform_filter(p.astype(np.float32), size=int(round(km * 1000 / RES)) | 1, mode="nearest")
temp_s = sm(temp)
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
byd = {d: [(a, b) for (dd, a, b) in PD if dd == d] for d in np.unique(grp)}
ALL = [(a, b) for (_, a, b) in PD]
def conc(p, pairs):
    C = Dd = 0.0
    for a, b in pairs:
        dp = p[a] - p[b]; dy = y[a] - y[b]
        if dp == 0: C += .5; Dd += .5
        elif np.sign(dp) == np.sign(dy): C += 1
        else: Dd += 1
    return C / (C + Dd)
base = temp_s[keep]; base_c = conc(base, ALL)
rng = np.random.RandomState(0); dists = np.unique(grp)
def P_vs_temp(p, reps=3000):
    diffs = []
    for _ in range(reps):
        ds = rng.choice(dists, len(dists), True); pr = [pp for d in ds for pp in byd[d]]
        diffs.append(conc(p, pr) - conc(base, pr))
    d = np.array(diffs); return d.mean(), (d > 0).mean()
print(f"temperature baseline concordance = {base_c:.3f}\n")
print(f"{'focal radius (hab & built)':30}{'concord':>9}{'gain':>8}{'P(>temp)':>10}")
best = None
for km in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    hab = sm(np.log1p(focal(np.maximum(rice_p, wet_p), km)))
    blt = sm(np.log1p(focal(built_p, km)))
    X = np.column_stack([temp_s, hab, blt])[keep]; p = oof(X); c = conc(p, ALL); m, P = P_vs_temp(p)
    print(f"{str(km)+' km':30}{c:>9.3f}{c-base_c:>+8.3f}{P:>10.3f}")
    if best is None or P > best[2]: best = (km, c, P)
print(f"\nbest by reliability: radius {best[0]} km  concordance {best[1]:.3f}  P(>temp)={best[2]:.3f}")
print("target: P(>temp) >= 0.95")
