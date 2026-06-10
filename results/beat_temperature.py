"""Hunt for a risk map that CLEARLY beats temperature on within-district concordance.

Uses clean (non-denominator-entangled) covariates with within-district signal:
rice_frac, built_frac, wetland_frac, dist_water_km, elevation, cropland_frac.
Fits with the WITHIN-estimator (district-demeaned) so the model learns local
relationships, and also pooled. Scores within-district concordance via LODO OOF,
and runs a PAIRED district-bootstrap of (model - temperature) to test significance.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

risk = pd.read_csv("results/rwanda_sector_risk_vs_incidence.csv"); risk["location_id"] = risk.location_id.astype(str)
terr = pd.read_csv("results/rwanda_sector_terrain.csv"); terr["location_id"] = terr.location_id.astype(str)
mo = pd.read_csv("results/rwanda_sector_moisture.csv"); mo["location_id"] = mo.location_id.astype(str)
lc = pd.read_csv("results/rwanda_sector_landcover_extra.csv"); lc["location_id"] = lc.location_id.astype(str)
df = (risk.merge(terr, on="location_id", how="left")
          .merge(mo[["location_id", "temp_pop", "ndvi_ann", "ndvi_amp", "ndvi_drop"]], on="location_id", how="left")
          .merge(lc, on="location_id", how="left"))
gdf = prepare_boundaries("RWA", 5)[["location_id", "geometry"]].copy(); gdf["location_id"] = gdf.location_id.astype(str)
gdf["area_km2"] = gdf.to_crs(32735).geometry.area / 1e6
df = df.merge(gdf[["location_id", "area_km2"]], on="location_id", how="left")
df["log_popdens"] = np.log1p(df["population"] / df["area_km2"])

y = df["annual_incidence_per1000"].values
g = df["district"].astype(str).values
def feats(cols):
    X = df[cols].values.astype(float)
    return np.where(np.isfinite(X), X, np.nanmean(np.where(np.isfinite(X), X, np.nan), 0))

logo = LeaveOneGroupOut()
def demean_fit_pred(model, X, tr, te):
    gtr = g[tr]; Xtr = X[tr].astype(float).copy(); ytr = y[tr].astype(float).copy()
    for d in np.unique(gtr):
        m = gtr == d; Xtr[m] -= Xtr[m].mean(0); ytr[m] -= ytr[m].mean()
    mdl = clone(model).fit(Xtr, ytr)
    Xte = X[te] - X[te].mean(0)
    return mdl.predict(Xte)
def oof(model, X, within):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, g):
        pred[te] = demean_fit_pred(model, X, tr, te) if within else clone(model).fit(X[tr], y[tr]).predict(X[te])
    return pred

def pair_lists():
    pl = []
    for d in np.unique(g):
        i = np.where(g == d)[0]
        for a in range(len(i)):
            for b in range(a + 1, len(i)):
                if y[i[a]] != y[i[b]]: pl.append((d, i[a], i[b]))
    return pl
PAIRSD = pair_lists()
PAIRS = [(a, b) for (_, a, b) in PAIRSD]
def concord_from_pairs(p, pairs):
    C = Dd = 0.0
    for a, b in pairs:
        dy = y[a] - y[b]; dp = p[a] - p[b]
        if dp == 0: C += .5; Dd += .5
        elif np.sign(dp) == np.sign(dy): C += 1
        else: Dd += 1
    return C / (C + Dd)

RF = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=0, n_jobs=-1)
SETS = {
    "temperature (baseline)":             (["temp_pop"], False),
    "temp + rice":                        (["temp_pop", "rice_frac"], True),
    "temp + rice + built":                (["temp_pop", "rice_frac", "built_frac"], True),
    "temp + rice + built + wetland + distW + elev": (
        ["temp_pop", "rice_frac", "built_frac", "wetland_frac", "dist_water_km", "elevation_m"], True),
    "[RF] temp + clean habitat set":      (
        ["temp_pop", "rice_frac", "built_frac", "wetland_frac", "dist_water_km", "elevation_m", "cropland_frac"], True),
    "temp + rice + built + log_popdens (denom-risky)": (
        ["temp_pop", "rice_frac", "built_frac", "log_popdens"], True),
}
print(f"sectors={len(y)} districts={pd.Series(g).nunique()} comparable-pairs={len(PAIRS)}\n")
base = oof(LinearRegression(), feats(["temp_pop"]), False)
base_c = concord_from_pairs(base, PAIRS)
print(f"{'model':50}{'concordance':>12}{'d vs temp':>11}{'P(>temp)':>10}")
preds = {}
for name, (cols, within) in SETS.items():
    mdl = RF if name.startswith("[RF]") else LinearRegression()
    p = oof(mdl, feats(cols), within); preds[name] = p
    c = concord_from_pairs(p, PAIRS)
    # paired district bootstrap of (model - temp)
    rng = np.random.RandomState(0); dists = np.unique(g); diffs = []
    by_d = {d: [(a, b) for (dd, a, b) in PAIRSD if dd == d] for d in dists}
    for _ in range(500):
        ds = rng.choice(dists, len(dists), replace=True)
        pr = [pp for d in ds for pp in by_d[d]]
        diffs.append(concord_from_pairs(p, pr) - concord_from_pairs(base, pr))
    diffs = np.array(diffs)
    print(f"{name:50}{c:>12.3f}{c-base_c:>+11.3f}{(diffs>0).mean():>10.2f}")
print("\n'd vs temp' = concordance gain over temperature; P(>temp) = paired-bootstrap prob model beats temp.")
print("CLEAR win = gain beyond ~0.03 and P(>temp) ~>=0.95.")
