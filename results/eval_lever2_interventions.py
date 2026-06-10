"""Lever 2 - interventions (bednets / IRS) as predictors of realised burden.
Add per-sector LLIN distribution + IRS coverage to the static map; burden-capture;
paired bootstrap. Discuss the targeting confound.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
foun = pd.read_csv("results/foundation_features.csv"); foun["location_id"] = foun.location_id.astype(str)
pan = pd.read_csv("data/inputs/chap_data_level5_clean_2013-2021.csv").rename(columns={"location": "location_id"})
pan["location_id"] = pan["location_id"].astype(str)
agg = pan.groupby("location_id").agg(
    llins_mass=("llins_mass_quantity_dispensed", "sum"),
    llins_epi=("llins_epi_quantity_dispensed", "sum"),
    irs_cov=("irs_spraying_coverage", "mean"),
    irs_prot=("irs_population_protected", "sum"),
    irs_months=("irs_sector_covered", "mean"),
).reset_index()
df = foun.merge(agg, on="location_id", how="left")
df["llins_pc"] = (df["llins_mass"].fillna(0) + df["llins_epi"].fillna(0)) / df["wp_pop"].replace(0, np.nan)
df["irs_prot_pc"] = df["irs_prot"].fillna(0) / df["wp_pop"].replace(0, np.nan)
for c in ["irs_cov", "irs_months"]: df[c] = df[c].fillna(0)
df = df[df.parent.notna() & df.inc_wp_w.notna() & df.wp_pop.gt(0)].reset_index(drop=True)
INTERV = ["llins_pc", "irs_cov", "irs_prot_pc", "irs_months"]
print("intervention coverage (per sector):")
print(df[INTERV].describe().round(3).to_string())
print("\nraw spatial corr with cleaned incidence:")
for c in INTERV: print(f"  {c:14}{spearmanr(df[c], df.inc_wp_w).statistic:+.3f}")

y = df["inc_wp_w"].values; grp = df["parent"].astype(str).values; pop = df["wp_pop"].values; cases = df["cases"].values
logo = LeaveOneGroupOut()
def oof(cols):
    X = df[cols].fillna(df[cols].mean()).values; p = np.full(len(df), np.nan)
    for tr, te in logo.split(X, y, grp): p[te] = clone(LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
    return p
def cap(sc, frac=0.5): o = np.argsort(-sc); cp = np.concatenate([[0], np.cumsum(pop[o]) / pop.sum()]); cc = np.concatenate([[0], np.cumsum(cases[o]) / cases.sum()]); return np.interp(frac, cp, cc)
rng = np.random.RandomState(0); dists = np.unique(grp); idxby = {d: np.where(grp == d)[0] for d in dists}
def pboot(a, b, frac=0.5):
    v = []
    for _ in range(1500):
        ds = rng.choice(dists, len(dists), True); idx = np.concatenate([idxby[d] for d in ds]); P = pop[idx]; Cc = cases[idx]
        def c(sc): o = np.argsort(-sc[idx]); cp = np.concatenate([[0], np.cumsum(P[o]) / P.sum()]); cc = np.concatenate([[0], np.cumsum(Cc[o]) / Cc.sum()]); return np.interp(frac, cp, cc)
        v.append(c(a) - c(b))
    return np.array(v)
BASE = ["sig_temp", "hab", "built"]
base = oof(BASE); oracle = cases / pop
print(f"\nburden@50%: baseline(static covars)={cap(base):.3f}  oracle={cap(oracle):.3f}")
for name, cols in [("+ LLIN only", BASE + ["llins_pc"]), ("+ IRS only", BASE + ["irs_cov", "irs_prot_pc", "irs_months"]),
                   ("+ all interventions", BASE + INTERV)]:
    p = oof(cols); b = pboot(p, base)
    print(f"  {name:22}{cap(p):.3f}   gap {b.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]  P={(b>0).mean():.3f}")
df[["location_id", "parent"] + INTERV].to_csv("results/sector_interventions.csv", index=False)
print("\nwrote results/sector_interventions.csv")
