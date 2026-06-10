"""Does a sigmoid temperature transform improve the gridded risk map's POOLED
(overall, between+within) ranking skill? Within-district temperature is ~constant
so a sigmoid is locally linear there (no gain); but pooled, temperature spans
14-25C and the threshold shape matters (report: logistic ~23C beat linear).

Pooled fit (LODO), pooled concordance over ALL sector pairs + pooled Spearman.
Sweep sigmoid (T0,k) vs linear temperature, holding habitat+built fixed, and also
temperature-only. Paired district-bootstrap for reliability.
"""
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
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
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and psum[i] > 0 for i in range(NS)])
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]]); grp = np.array([par[loc[i]] for i in range(NS) if keep[i]])
T, H, B = temp_s[keep], hab_s[keep], built_s[keep]
logo = LeaveOneGroupOut()
def oof_pooled(X):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, grp):
        pred[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    return pred
# ALL sector pairs (pooled concordance)
ALL = [(a, b) for a in range(len(y)) for b in range(a + 1, len(y)) if y[a] != y[b]]
def conc(p, pairs):
    C = Dd = 0.0
    for a, b in pairs:
        dp = p[a] - p[b]; dy = y[a] - y[b]
        if dp == 0: C += .5; Dd += .5
        elif np.sign(dp) == np.sign(dy): C += 1
        else: Dd += 1
    return C / (C + Dd)

def sigT(T0, k): return sec_mean(expit((temp - T0) / k))[keep]

# baselines
lin_full = oof_pooled(np.column_stack([T, H, B])); c_lin = conc(lin_full, ALL); r_lin = spearmanr(lin_full, y).statistic
lin_tonly = oof_pooled(T[:, None]); c_t = conc(lin_tonly, ALL); r_t = spearmanr(lin_tonly, y).statistic
print(f"POOLED skill (concordance / Spearman):")
print(f"  raw temperature only         : {c_t:.3f} / {r_t:.3f}")
print(f"  linear-temp gridded (+hab+blt): {c_lin:.3f} / {r_lin:.3f}\n")

print(f"{'sigmoid (full model)':24}{'concord':>9}{'Spearman':>10}{'vs linear':>11}")
res = []
for T0 in [20, 21, 22, 23, 24, 25]:
    for k in [0.5, 1.0, 1.5, 2.0, 3.0]:
        st = sigT(T0, k)
        p = oof_pooled(np.column_stack([st, H, B])); c = conc(p, ALL)
        res.append((T0, k, c, spearmanr(p, y).statistic, p))
res.sort(key=lambda r: -r[2])
for T0, k, c, r, _ in res[:8]:
    print(f"  T0={T0} k={k:<4}            {c:>9.3f}{r:>10.3f}{c-c_lin:>+11.3f}")
bT0, bk, bc, br, bp = res[0]

rng = np.random.RandomState(0); dists = np.unique(grp)
byd = {d: [(a, b) for (a, b) in ALL if grp[a] == d or grp[b] == d] for d in dists}
def paired(p, ref):
    diffs = []
    for _ in range(1500):
        ds = set(rng.choice(dists, len(dists), True))
        pr = [(a, b) for (a, b) in ALL if grp[a] in ds and grp[b] in ds]
        diffs.append(conc(p, pr) - conc(ref, pr))
    d = np.array(diffs); return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()
m1, ci1, P1 = paired(bp, lin_full)
# temperature-only: sigmoid vs linear
st = sigT(bT0, bk); ptonly = oof_pooled(st[:, None]); c_sig_t = conc(ptonly, ALL)
m2, ci2, P2 = paired(ptonly, lin_tonly)
print(f"\nbest sigmoid (full model): T0={bT0}, k={bk}  concordance={bc:.3f}  Spearman={br:.3f}")
print(f"  vs linear-temp gridded : gain {m1:+.3f}  CI[{ci1[0]:+.3f},{ci1[1]:+.3f}]  P(>linear)={P1:.3f}")
print(f"\ntemperature-ONLY, same sigmoid vs linear temp:")
print(f"  sigmoid temp-only concordance={c_sig_t:.3f} (linear temp-only {c_t:.3f})")
print(f"  gain {m2:+.3f}  CI[{ci2[0]:+.3f},{ci2[1]:+.3f}]  P(>linear)={P2:.3f}")
