"""Final: select the best within-district land-use model and test it against
temperature with a high-resolution PAIRED district-bootstrap (gain, 95% CI,
one-sided P). Save per-sector scores and render the risk map.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import chap_gis as cgis
from chap_gis.cli.dynamic import prepare_boundaries

risk = pd.read_csv("results/rwanda_sector_risk_vs_incidence.csv"); risk["location_id"] = risk.location_id.astype(str)
terr = pd.read_csv("results/rwanda_sector_terrain.csv"); terr["location_id"] = terr.location_id.astype(str)
mo = pd.read_csv("results/rwanda_sector_moisture.csv"); mo["location_id"] = mo.location_id.astype(str)
lc = pd.read_csv("results/rwanda_sector_landcover_extra.csv"); lc["location_id"] = lc.location_id.astype(str)
df = (risk.merge(terr, on="location_id").merge(mo[["location_id", "temp_pop"]], on="location_id").merge(lc, on="location_id"))
df["rice_log"] = np.log1p(df.rice_frac); df["built_log"] = np.log1p(df.built_frac)
df["wetland_log"] = np.log1p(df.wetland_frac); df["habitat_log"] = np.log1p(df.rice_frac + df.wetland_frac)
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
CANDS = {
    "temp + habitat_log + built_log":            ["temp_pop", "habitat_log", "built_log"],
    "temp + rice_log + built_log":               ["temp_pop", "rice_log", "built_log"],
    "temp + rice_log + wetland_log + built_log": ["temp_pop", "rice_log", "wetland_log", "built_log"],
}
rng = np.random.RandomState(0); dists = np.unique(g)
print(f"temp baseline within-district concordance = {base_c:.3f}\n")
print(f"{'model':46}{'concord':>9}{'gain':>8}{'95% CI':>18}{'P(>temp)':>10}")
best = None
for name, cols in CANDS.items():
    p = oof(feats(cols), True); c = conc(p, ALL)
    diffs = []
    for _ in range(2000):
        ds = rng.choice(dists, len(dists), replace=True)
        pr = [pp for d in ds for pp in by_d[d]]
        diffs.append(conc(p, pr) - conc(base, pr))
    diffs = np.array(diffs); lo, hi = np.percentile(diffs, [2.5, 97.5]); P = (diffs > 0).mean()
    print(f"{name:46}{c:>9.3f}{c-base_c:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{P:>10.3f}")
    if best is None or c > best[1]: best = (name, c, cols, p, (lo, hi, P))
print(f"\nbest: {best[0]}  concordance={best[1]:.3f}")

# per-sector scores: between-district level (temp, pooled) + within-district adj (best, within)
df["risk_within"] = best[3]
df["temp_rank_pred"] = base
df[["location_id", "district", "annual_incidence_per1000", "temp_pop",
    "risk_within", "rice_frac", "built_frac", "wetland_frac"]].to_csv(
    "results/within_district_best_model.csv", index=False)

# render: choropleth of within-district risk adjustment for the winning model
gdf = prepare_boundaries("RWA", 5)[["location_id", "geometry"]].copy(); gdf["location_id"] = gdf.location_id.astype(str)
gdf["geometry"] = gdf.geometry.make_valid()
m = gdf.merge(df[["location_id", "risk_within", "annual_incidence_per1000"]], on="location_id")
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
m.plot(ax=ax[0], column="risk_within", cmap="RdYlGn_r", legend=True, edgecolor="grey", linewidth=.2,
       legend_kwds={"label": "within-district risk score", "shrink": .6})
ax[0].set_title(f"Winning within-district risk model\n{best[0]}  (concordance {best[1]:.3f} vs temp {base_c:.3f})", fontsize=11)
ax[0].axis("off")
vmax = np.nanpercentile(m["annual_incidence_per1000"], 95)
m.plot(ax=ax[1], column="annual_incidence_per1000", cmap="RdYlGn_r", vmin=0, vmax=vmax,
       legend=True, edgecolor="grey", linewidth=.2, legend_kwds={"label": "incidence /1000", "shrink": .6})
ax[1].set_title("Observed incidence (quantiles)", fontsize=11); ax[1].axis("off")
fig.tight_layout(); fig.savefig("results/within_district_best_model.png", dpi=150, bbox_inches="tight")
print("wrote results/within_district_best_model.{csv,png}")
