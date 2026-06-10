"""Does the spatial-GP burden-capture gain survive SPATIALLY-BUFFERED CV?
Standard LODO holds out contiguous districts but leaves their immediate neighbours
in training; a GP can exploit that proximity. Re-evaluate the spatial add-on while
excluding training sectors within a buffer (km) of the held-out district.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.base import clone
df = pd.read_csv("results/foundation_features.csv"); df["location_id"] = df.location_id.astype(str)
df = df[df.parent.notna() & df.inc_wp_w.notna() & df.wp_pop.gt(0)].reset_index(drop=True)
y = df["inc_wp_w"].values; grp = df["parent"].astype(str).values
pop = df["wp_pop"].values; cases = df["cases"].values
BASE = ["sig_temp", "hab", "built"]; X = df[BASE].values
lat0 = df.lat.mean(); kmx = 111.0 * np.cos(np.radians(lat0)); kmy = 111.0
cx = df.lon.values * kmx; cy = df.lat.values * kmy; co = np.column_stack([cx, cy])
dists = np.unique(grp)
def dmat(A, B): return np.sqrt(((A[:, None, :] - B[None, :, :])**2).sum(2))

def run(buffer_km, spatial):
    pred = np.full(len(df), np.nan)
    for d in dists:
        te = np.where(grp == d)[0]; others = np.where(grp != d)[0]
        if buffer_km > 0:
            dd = dmat(co[others], co[te]).min(1)
            tr = others[dd > buffer_km]
        else:
            tr = others
        m = clone(LinearRegression()).fit(X[tr], y[tr]); p = m.predict(X[te])
        if spatial:
            res = y[tr] - m.predict(X[tr])
            k = C(np.var(res)) * RBF(20.0, (5.0, 200.0)) + WhiteKernel(np.var(res))
            gp = GaussianProcessRegressor(kernel=k, alpha=1e-6).fit(co[tr], res)
            p = p + gp.predict(co[te])
        pred[te] = p
    return pred
def cap(sc, frac=0.5):
    o = np.argsort(-sc); cp = np.concatenate([[0], np.cumsum(pop[o]) / pop.sum()]); cc = np.concatenate([[0], np.cumsum(cases[o]) / cases.sum()]); return np.interp(frac, cp, cc)
rng = np.random.RandomState(0); idxby = {d: np.where(grp == d)[0] for d in dists}
def pboot(a, b):
    v = []
    for _ in range(1500):
        ds = rng.choice(dists, len(dists), True); idx = np.concatenate([idxby[d] for d in ds]); P = pop[idx]; Cc = cases[idx]
        def c(sc): o = np.argsort(-sc[idx]); cp = np.concatenate([[0], np.cumsum(P[o]) / P.sum()]); cc = np.concatenate([[0], np.cumsum(Cc[o]) / Cc.sum()]); return np.interp(.5, cp, cc)
        v.append(c(a) - c(b))
    return np.array(v)
print(f"{'buffer':>8}{'base':>8}{'+spatial':>10}{'gain':>8}{'P(spatial>base)':>17}{'min train n':>12}")
for bkm in [0, 10, 20, 30]:
    base = run(bkm, False); sp = run(bkm, True); b = pboot(sp, base)
    minn = min((grp != d).sum() if bkm == 0 else (dmat(co[np.where(grp != d)[0]], co[np.where(grp == d)[0]]).min(1) > bkm).sum() for d in dists)
    print(f"{str(bkm)+'km':>8}{cap(base):>8.3f}{cap(sp):>10.3f}{cap(sp)-cap(base):>+8.3f}{(b>0).mean():>17.3f}{minn:>12}")
print("\ngain shrinking toward 0 as buffer grows => the spatial gain was proximity-driven")
