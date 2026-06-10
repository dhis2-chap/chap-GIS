"""Re-test temp vs land-use models after excluding the 8 documented denominator
artifacts (incidence > 1000/1000 facility catchments). Within-estimator,
within-district concordance, paired district-bootstrap vs temperature.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone

risk = pd.read_csv("results/rwanda_sector_risk_vs_incidence.csv"); risk["location_id"] = risk.location_id.astype(str)
terr = pd.read_csv("results/rwanda_sector_terrain.csv"); terr["location_id"] = terr.location_id.astype(str)
mo = pd.read_csv("results/rwanda_sector_moisture.csv"); mo["location_id"] = mo.location_id.astype(str)
lc = pd.read_csv("results/rwanda_sector_landcover_extra.csv"); lc["location_id"] = lc.location_id.astype(str)
df = (risk.merge(terr, on="location_id", how="left")
          .merge(mo[["location_id", "temp_pop"]], on="location_id", how="left")
          .merge(lc, on="location_id", how="left"))
df = df[df["annual_incidence_per1000"] <= 1000].reset_index(drop=True)        # drop facility-catchment artifacts
for c in ["rice_frac", "built_frac", "wetland_frac"]:
    df[c + "_log"] = np.log1p(df[c])
df["habitat_log"] = np.log1p(df["rice_frac"] + df["wetland_frac"])

y = df["annual_incidence_per1000"].values; g = df["district"].astype(str).values
def feats(cols):
    X = df[cols].values.astype(float)
    return np.where(np.isfinite(X), X, np.nanmean(np.where(np.isfinite(X), X, np.nan), 0))
logo = LeaveOneGroupOut()
def oof(X, within):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, g):
        if within:
            gtr = g[tr]; Xtr = X[tr].astype(float).copy(); ytr = y[tr].astype(float).copy()
            for d in np.unique(gtr):
                m = gtr == d; Xtr[m] -= Xtr[m].mean(0); ytr[m] -= ytr[m].mean()
            pred[te] = clone(LinearRegression()).fit(Xtr, ytr).predict(X[te] - X[te].mean(0))
        else:
            pred[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    return pred
PD = []
for d in np.unique(g):
    i = np.where(g == d)[0]
    for a in range(len(i)):
        for b in range(a + 1, len(i)):
            if y[i[a]] != y[i[b]]: PD.append((d, i[a], i[b]))
by_d = {d: [(a, b) for (dd, a, b) in PD if dd == d] for d in np.unique(g)}
ALL = [(a, b) for (_, a, b) in PD]
def conc(p, pairs):
    C = Dd = 0.0
    for a, b in pairs:
        dp = p[a] - p[b]; dy = y[a] - y[b]
        if dp == 0: C += .5; Dd += .5
        elif np.sign(dp) == np.sign(dy): C += 1
        else: Dd += 1
    return C / (C + Dd)

base = oof(feats(["temp_pop"]), False); base_c = conc(base, ALL)
SETS = {
    "temp + rice + built":             ["temp_pop", "rice_frac", "built_frac"],
    "temp + rice_log + built_log":     ["temp_pop", "rice_frac_log", "built_frac_log"],
    "temp + habitat_log + built_log":  ["temp_pop", "habitat_log", "built_frac_log"],
}
print(f"sectors={len(df)} (excluded >1000/1000)  districts={pd.Series(g).nunique()}  pairs={len(ALL)}")
print(f"temp baseline concordance={base_c:.3f}\n")
print(f"{'within-estimator model':36}{'concord':>9}{'d vs temp':>11}{'P(>temp)':>10}")
rng = np.random.RandomState(0); dists = np.unique(g)
for name, cols in SETS.items():
    p = oof(feats(cols), True); c = conc(p, ALL); diffs = []
    for _ in range(800):
        ds = rng.choice(dists, len(dists), replace=True)
        pr = [pp for d in ds for pp in by_d[d]]
        diffs.append(conc(p, pr) - conc(base, pr))
    diffs = np.array(diffs)
    print(f"{name:36}{c:>9.3f}{c-base_c:>+11.3f}{(diffs>0).mean():>10.2f}")
print("\n(vs full-data temp baseline 0.684; clean run isolates rankable signal)")
