"""Push the gridded within-district concordance to a reliable improvement.

Adds a griddable urbanization signal -- focal log population intensity (the
strongest within-district covariate in the screen, -0.18 partial | temp, and
NOT a denominator artifact within districts: cases~pop slope = 1.0). Search
feature sets / focal radii; within-estimator concordance + paired district
bootstrap vs temperature (target P>=0.95).
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
hab = lambda km: sm(np.log1p(focal(np.maximum(rice_p, wet_p), km)))
blt = lambda km: sm(np.log1p(focal(built_p, km)))
popd = lambda km: sm(np.log1p(focal(pop, km)))          # focal population intensity (urbanisation)

tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and psum[i] > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]]); grp = np.array([par[loc[i]] for i in range(NS) if keep[i]])
logo = LeaveOneGroupOut()
def oof_within(X):
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
def P_vs_temp(p):
    diffs = []
    for _ in range(2000):
        ds = rng.choice(dists, len(dists), True); pr = [pp for d in ds for pp in byd[d]]
        diffs.append(conc(p, pr) - conc(base, pr))
    d = np.array(diffs); return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()

print(f"temperature baseline concordance = {base_c:.3f}\n")
print(f"{'gridded model':46}{'concord':>9}{'gain':>8}{'P(>temp)':>10}")
SETS = {
    "temp + hab.5 + built.5 (current)":            [temp_s, hab(0.5), blt(0.5)],
    "temp + hab.5 + built.5 + popdens1":           [temp_s, hab(0.5), blt(0.5), popd(1.0)],
    "temp + hab.5 + popdens1":                     [temp_s, hab(0.5), popd(1.0)],
    "temp + hab.5 + built.5 + popdens.5":          [temp_s, hab(0.5), blt(0.5), popd(0.5)],
    "temp + hab1 + built.5 + popdens1":            [temp_s, hab(1.0), blt(0.5), popd(1.0)],
    "temp + hab.5 + popdens2":                     [temp_s, hab(0.5), popd(2.0)],
}
res = []
for name, fl in SETS.items():
    X = np.column_stack(fl)[keep]; p = oof_within(X); c = conc(p, ALL); m, ci, P = P_vs_temp(p)
    res.append((name, c, P, p)); print(f"{name:46}{c:>9.3f}{c-base_c:>+8.3f}{P:>10.3f}   CI[{ci[0]:+.3f},{ci[1]:+.3f}]")
print("\ntarget: concordance > 0.706 and P(>temp) >= 0.95")
