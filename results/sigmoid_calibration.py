"""Ranking is invariant to a monotonic (sigmoid) temperature transform, so a
sigmoid can only improve the gridded map's CALIBRATION (predicted incidence
values). The temperature-incidence relationship is nonlinear (report: threshold
~23C), so a sigmoid should fit better than raw-linear temperature.

LODO out-of-fold R2 and RMSE; linear vs sigmoid temperature, temp-only and full
model; sweep (T0,k); paired district-bootstrap of RMSE improvement.
Excludes the 8 documented denominator artifacts (>1000/1000) for a stable fit.
"""
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

z = np.load("results/_gridded_arrays.npz", allow_pickle=True)
temp, fhL, fbL, pop, sect = z["temp"], z["fhL"], z["fbL"], z["pop"], z["sect"]
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
T, H, B = temp_s[keep], hab_s[keep], built_s[keep]
logo = LeaveOneGroupOut()
def oof(X):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, grp):
        pred[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    return pred
def r2(p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
def rmse(p): return np.sqrt(np.mean((y - p) ** 2))
def sigT(T0, k): return sec_mean(expit((temp - T0) / k))[keep]

lin_t = oof(T[:, None]); lin_f = oof(np.column_stack([T, H, B]))
print(f"n={len(y)} (excl >1000/1000)   metric: LODO R2 / RMSE")
print(f"  linear temp only : R2={r2(lin_t):+.3f}  RMSE={rmse(lin_t):.1f}")
print(f"  linear full      : R2={r2(lin_f):+.3f}  RMSE={rmse(lin_f):.1f}\n")

print(f"{'sigmoid':16}{'R2(t-only)':>12}{'R2(full)':>10}{'RMSE(full)':>12}")
res = []
for T0 in [19, 20, 21, 22, 23, 24]:
    for k in [0.5, 1.0, 1.5, 2.0, 3.0]:
        st = sigT(T0, k)
        pf = oof(np.column_stack([st, H, B])); pt = oof(st[:, None])
        res.append((T0, k, r2(pt), r2(pf), rmse(pf), pf))
res.sort(key=lambda r: -r[3])
for T0, k, rt, rf, rm, _ in res[:8]:
    print(f"  T0={T0} k={k:<4}     {rt:>12.3f}{rf:>10.3f}{rm:>12.1f}")
bT0, bk, brt, brf, brm, bpf = res[0]
# best temp-only sigmoid (isolates the response-shape effect)
rest = sorted(res, key=lambda r: -r[2])[0]
print(f"\nbest full sigmoid: T0={bT0}, k={bk}  R2(full)={brf:.3f} (linear {r2(lin_f):.3f})  RMSE {brm:.1f} (linear {rmse(lin_f):.1f})")
print(f"best temp-only sigmoid: T0={rest[0]}, k={rest[1]}  R2(t-only)={rest[2]:.3f} (linear {r2(lin_t):.3f})")

rng = np.random.RandomState(0); dists = np.unique(grp)
def paired_rmse(p, ref):
    d = []
    for _ in range(2000):
        ds = rng.choice(dists, len(dists), True)
        idx = np.concatenate([np.where(grp == dd)[0] for dd in ds])
        d.append(np.sqrt(np.mean((y[idx] - ref[idx]) ** 2)) - np.sqrt(np.mean((y[idx] - p[idx]) ** 2)))
    d = np.array(d); return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()   # >0 => sigmoid lower RMSE
m, ci, P = paired_rmse(bpf, lin_f)
ptbest = oof(sigT(rest[0], rest[1])[:, None]); m2, ci2, P2 = paired_rmse(ptbest, lin_t)
print(f"\nfull model, sigmoid vs linear RMSE improvement: {m:+.1f}  CI[{ci[0]:+.1f},{ci[1]:+.1f}]  P(sigmoid better)={P:.3f}")
print(f"temp-only,  sigmoid vs linear RMSE improvement: {m2:+.1f}  CI[{ci2[0]:+.1f},{ci2[1]:+.1f}]  P(sigmoid better)={P2:.3f}")
