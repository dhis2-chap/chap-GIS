"""Evaluate static-risk-map levers on the headline burden-capture metric.
Lever 6 (denominators), 1 (spatial), 4 (hydrology), 5 (urban) -> combined.
All LODO out-of-fold; paired district-bootstrap vs the cleaned baseline.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
df = pd.read_csv("results/foundation_features.csv"); df["location_id"] = df.location_id.astype(str)
grp = df["parent"].astype(str).values; cases = df["cases"].values
wp = df["wp_pop"].values; dh = df["dhis2_pop"].values
BASE = ["sig_temp", "hab", "built"]; HYDRO = ["valley", "twi", "slope", "water", "wetland"]; URBAN = ["logpopdens"]
logo = LeaveOneGroupOut()
def oof(cols, y, spatial=False):
    X = df[cols].values; p = np.full(len(df), np.nan)
    for tr, te in logo.split(X, y, grp):
        m = clone(LinearRegression()).fit(X[tr], y[tr]); p[te] = m.predict(X[te])
        if spatial:
            res = y[tr] - m.predict(X[tr]); co = df[["lon", "lat"]].values
            k = C(np.var(res)) * RBF(0.15, (0.03, 1.0)) + WhiteKernel(np.var(res))
            gp = GaussianProcessRegressor(kernel=k, normalize_y=False, alpha=1e-6).fit(co[tr], res)
            p[te] += gp.predict(co[te])
    return p
def cap(scores, pop, frac=0.5):
    o = np.argsort(-scores); cp = np.concatenate([[0], np.cumsum(pop[o]) / pop.sum()]); cc = np.concatenate([[0], np.cumsum(cases[o]) / cases.sum()])
    return np.interp(frac, cp, cc)
rng = np.random.RandomState(0); dists = np.unique(grp); idxby = {d: np.where(grp == d)[0] for d in dists}
def pboot(scoreA, scoreB, pop, frac=0.5, reps=1500):
    v = []
    for _ in range(reps):
        ds = rng.choice(dists, len(dists), True); idx = np.concatenate([idxby[d] for d in ds]); P = pop[idx]
        def c(sc): o = np.argsort(-sc[idx]); cp = np.concatenate([[0], np.cumsum(P[o]) / P.sum()]); cc = np.concatenate([[0], np.cumsum(cases[idx][o]) / cases[idx].sum()]); return np.interp(frac, cp, cc)
        v.append(c(scoreA) - c(scoreB))
    return np.array(v)

yd = df["inc_dhis2"].values; yw = df["inc_wp_w"].values
oracle_wp = cases / wp; oracle_dh = cases / dh
base_dh = oof(BASE, yd)
print("=== LEVER 6: denominators ===")
print(f"  baseline, pop=DHIS2  : burden@50%={cap(base_dh, dh):.3f}   oracle={cap(oracle_dh, dh):.3f}")
print(f"  baseline, pop=WorldPop: burden@50%={cap(base_dh, wp):.3f}   oracle={cap(oracle_wp, wp):.3f}")
base_w = oof(BASE, yw)
print(f"  baseline trained on cleaned target (inc_wp_w), pop=WorldPop: burden@50%={cap(base_w, wp):.3f}")
print("  -> going forward: pop=WorldPop, target=inc_wp_w (cleaned)\n")

y = yw; pop = wp; base = base_w; base_c = cap(base, pop)
print(f"=== levers vs cleaned baseline (burden@50%, pop=WorldPop)  baseline={base_c:.3f}  oracle={cap(oracle_wp,pop):.3f} ===")
def line(name, scores):
    b = pboot(scores, base, pop); print(f"  {name:34}{cap(scores,pop):.3f}   gap {b.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]  P={(b>0).mean():.3f}")
    return scores
sp = line("+ spatial (lever 1)", oof(BASE, y, spatial=True))
hy = line("+ hydrology (lever 4)", oof(BASE + HYDRO, y))
ur = line("+ urban/popdens (lever 5)", oof(BASE + URBAN, y))
comb_feats = line("+ hydro + urban (features)", oof(BASE + HYDRO + URBAN, y))
combo = line("+ hydro + urban + SPATIAL (all)", oof(BASE + HYDRO + URBAN, y, spatial=True))
np.savez("results/_static_levers.npz", base=base, spatial=sp, hydro=hy, urban=ur, comb_feats=comb_feats, combo=combo,
         oracle=oracle_wp, pop=pop, cases=cases, base_dh=base_dh)
print("\nwrote results/_static_levers.npz")
