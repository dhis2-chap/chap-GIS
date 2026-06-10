"""Is population density a legitimate within-district predictor (urbanization),
or a denominator artifact? Test cases~pop scaling, then use log pop-density.

If log(cases) scales ~linearly with log(pop) (slope ~1), incidence is not
mechanically driven by 1/pop, so pop-density's negative association is real
epidemiology (urban malaria is lower). Then test within-estimator models that
use it, vs temperature, with paired district-bootstrap concordance.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

risk = pd.read_csv("results/rwanda_sector_risk_vs_incidence.csv"); risk["location_id"] = risk.location_id.astype(str)
terr = pd.read_csv("results/rwanda_sector_terrain.csv"); terr["location_id"] = terr.location_id.astype(str)
mo = pd.read_csv("results/rwanda_sector_moisture.csv"); mo["location_id"] = mo.location_id.astype(str)
lc = pd.read_csv("results/rwanda_sector_landcover_extra.csv"); lc["location_id"] = lc.location_id.astype(str)
sw = pd.read_csv("results/rwanda_sweep_temp.csv"); sw["location_id"] = sw.location_id.astype(str)
cp = sw.groupby("location_id").agg(cases=("disease", "sum"), pop_health=("population", "mean")).reset_index()

df = (risk.merge(terr, on="location_id").merge(mo[["location_id", "temp_pop"]], on="location_id")
          .merge(lc, on="location_id").merge(cp, on="location_id", how="left"))
gdf = prepare_boundaries("RWA", 5)[["location_id", "geometry"]].copy(); gdf["location_id"] = gdf.location_id.astype(str)
gdf["area_km2"] = gdf.to_crs(32735).geometry.area / 1e6
df = df.merge(gdf[["location_id", "area_km2"]], on="location_id")
df["log_popdens"] = np.log1p(df["population"] / df["area_km2"])
df["rice_log"] = np.log1p(df["rice_frac"]); df["built_log"] = np.log1p(df["built_frac"])
df["habitat_log"] = np.log1p(df["rice_frac"] + df["wetland_frac"])
g = df["district"].astype(str).values

# --- scaling check: log(cases) ~ log(pop), overall and within-district ---
ok = (df["cases"] > 0) & (df["pop_health"] > 0)
lc_, lp = np.log(df.loc[ok, "cases"]), np.log(df.loc[ok, "pop_health"])
b_overall = np.polyfit(lp, lc_, 1)[0]
gg = g[ok.values]; lcd = lc_.values.copy(); lpd = lp.values.copy()
for d in np.unique(gg):
    m = gg == d; lcd[m] -= lcd[m].mean(); lpd[m] -= lpd[m].mean()
b_within = np.polyfit(lpd, lcd, 1)[0]
print(f"cases~pop scaling: overall slope={b_overall:.2f}  within-district slope={b_within:.2f}")
print("  (slope ~1 => incidence not mechanically ~1/pop => pop-density is a real covariate)\n")

y = df["annual_incidence_per1000"].values
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
PD = [(d, i[a], i[b]) for d in np.unique(g) for i in [np.where(g == d)[0]]
      for a in range(len(i)) for b in range(a + 1, len(i)) if y[i[a]] != y[i[b]]]
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
    "temp + rice_log + log_popdens":               ["temp_pop", "rice_log", "log_popdens"],
    "temp + habitat_log + log_popdens":            ["temp_pop", "habitat_log", "log_popdens"],
    "temp + habitat_log + built_log + log_popdens": ["temp_pop", "habitat_log", "built_log", "log_popdens"],
}
print(f"temp baseline concordance={base_c:.3f}\n")
print(f"{'within-estimator model':46}{'concord':>9}{'d vs temp':>11}{'P(>temp)':>10}")
rng = np.random.RandomState(0); dists = np.unique(g)
for name, cols in SETS.items():
    p = oof(feats(cols), True); c = conc(p, ALL); diffs = []
    for _ in range(800):
        ds = rng.choice(dists, len(dists), replace=True)
        pr = [pp for d in ds for pp in by_d[d]]
        diffs.append(conc(p, pr) - conc(base, pr))
    print(f"{name:46}{c:>9.3f}{c-base_c:>+11.3f}{(np.array(diffs)>0).mean():>10.2f}")
print("\nCLEAR win target: d vs temp >= ~0.03 and P(>temp) >= 0.95")
