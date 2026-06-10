"""Interaction effects among temperature / habitat / built-up.

Mechanism: breeding habitat raises risk only where temperature permits
transmission -> a temp x habitat interaction (the multiplicative S(T)*habitat
structure the additive model omits). Test temp x habitat, temp x built,
habitat x built on BOTH metrics:
  - calibration: pooled LODO R2 (sigmoid temp base, excl >1000 artifacts)
  - ranking: within-district concordance (within-estimator, temp base)
Features z-scored before forming products. Paired district-bootstrap for the
best interaction vs the additive model.
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
def sm(a): return np.bincount(s, weights=w * a[ok].astype(np.float64), minlength=NS) / np.maximum(psum, 1e-9)
temp_s, hab_s, blt_s = sm(temp), sm(fhL), sm(fbL)
sig_s = sm(expit((temp - 19.0) / 0.5))
tgt = pd.read_csv("results/rwanda_sector_control_temp.csv").set_index("location_id")["annual_incidence_per1000"]
g = cgis.io.boundaries.load("RWA", level=5)[["shapeID", "parent"]]; g["location_id"] = g.shapeID.astype(str)
par = g.set_index("location_id")["parent"]
keep = np.array([(loc[i] in tgt.index) and pd.notna(par.get(loc[i], np.nan)) and psum[i] > 0 for i in range(NS)])
keepc = keep & np.array([tgt.get(loc[i], 9e9) <= 1000 for i in range(NS)])
def zc(v): return (v - v.mean()) / v.std()
y = np.array([tgt[loc[i]] for i in range(NS) if keep[i]]); grp = np.array([par[loc[i]] for i in range(NS) if keep[i]])
yc = np.array([tgt[loc[i]] for i in range(NS) if keepc[i]]); grpc = np.array([par[loc[i]] for i in range(NS) if keepc[i]])
logo = LeaveOneGroupOut()

def cols(base_temp, mask):
    T = zc(base_temp)[mask]; H = zc(hab_s)[mask]; B = zc(blt_s)[mask]
    return dict(T=T, H=H, B=B, TH=T * H, TB=T * B, HB=H * B)

def oof_pool(X, yy, gg):
    p = np.full(len(yy), np.nan)
    for tr, te in logo.split(X, yy, gg): p[te] = clone(LinearRegression()).fit(X[tr], yy[tr]).predict(X[te])
    return p
def oof_within(X, yy, gg):
    p = np.full(len(yy), np.nan)
    for tr, te in logo.split(X, yy, gg):
        gt = gg[tr]; Xt = X[tr].copy(); yt = yy[tr].copy()
        for d in np.unique(gt):
            m = gt == d; Xt[m] -= Xt[m].mean(0); yt[m] -= yt[m].mean()
        p[te] = clone(LinearRegression()).fit(Xt, yt).predict(X[te] - X[te].mean(0))
    return p
def r2(p): return 1 - np.sum((yc - p) ** 2) / np.sum((yc - yc.mean()) ** 2)
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

cR = cols(sig_s, keepc)   # calibration uses sigmoid temp
cC = cols(temp_s, keep)   # ranking uses temp
def build(c, names): return np.column_stack([c[n] for n in names])
SETS = {
    "additive (T,H,B)":            ["T", "H", "B"],
    "+ T x habitat":               ["T", "H", "B", "TH"],
    "+ T x built":                 ["T", "H", "B", "TB"],
    "+ habitat x built":           ["T", "H", "B", "HB"],
    "+ all pairwise":              ["T", "H", "B", "TH", "TB", "HB"],
}
print("metric: calibration R2 (pooled, sigmoid temp) | within-district concordance (temp)\n")
print(f"{'model':28}{'calib R2':>10}{'concord':>10}")
preds_r = {}; preds_c = {}
for name, ns in SETS.items():
    pr = oof_pool(build(cR, ns), yc, grpc); pc = oof_within(build(cC, ns), y, grp)
    preds_r[name] = pr; preds_c[name] = pc
    print(f"{name:28}{r2(pr):>10.3f}{conc(pc):>10.3f}")

# paired bootstrap of the mechanistic T x habitat vs additive, both metrics
rng = np.random.RandomState(0)
distsc = np.unique(grpc); dists = np.unique(grp)
def pb_r2(p, ref):
    d = []
    for _ in range(2000):
        ds = rng.choice(distsc, len(distsc), True); idx = np.concatenate([np.where(grpc == dd)[0] for dd in ds])
        ssr = lambda q: np.sum((yc[idx] - q[idx]) ** 2)
        d.append(ssr(ref) - ssr(p))           # >0 => interaction lower SSE
    d = np.array(d); return (d > 0).mean()
byd = {d: [(a, b) for (a, b) in ALL if grp[a] == d] for d in dists}
def pb_conc(p, ref):
    d = []
    for _ in range(2000):
        ds = rng.choice(dists, len(dists), True); pr = [pp for dd in ds for pp in byd[dd]]
        cc = lambda q: (sum((np.sign(q[a]-q[b])==np.sign(y[a]-y[b])) for a,b in pr))/max(len(pr),1)
        d.append(cc(p) - cc(ref))
    return (np.array(d) > 0).mean()
print(f"\nT x habitat vs additive:")
print(f"  calibration: P(interaction better R2) = {pb_r2(preds_r['+ T x habitat'], preds_r['additive (T,H,B)']):.3f}")
print(f"  ranking:     P(interaction better concord) = {pb_conc(preds_c['+ T x habitat'], preds_c['additive (T,H,B)']):.3f}")
print("\nref additive: calib R2 0.402 (sigmoid) / concordance 0.706")
