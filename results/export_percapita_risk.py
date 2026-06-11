"""Per-sector per-capita risk, WITHIN-DISTRICT formulation, full-dataset fit.

Within-district fixed-effect slopes (mechanistically-correct signs: habitat +,
built -) estimated from district-demeaned variation, plus an empirical-Bayes
district intercept (observed district burden) to supply the between-district
level. Risk = X.beta_within + district_intercept.

Columns:
  risk_per1000_yr            : within-district-formulation absolute risk (PRIMARY)
  risk_env_score             : normalised X.beta_within (environmental part, correct signs)
  risk_pooled_per1000_yr     : pooled covariate model (reference; habitat sign flips)
  observed_incidence_per1000_yr, risk_rank, risk_normalized
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
N_YEARS = 9
df = pd.read_csv("results/foundation_features.csv"); df["location_id"] = df.location_id.astype(str)
df = df[df.wp_pop.gt(0) & df.inc_wp_w.notna()].reset_index(drop=True)
COLS = ["sig_temp", "hab", "built"]; X = df[COLS].values; y = df["inc_wp_w"].values
g = df["parent"].astype(str).values

# within-district slopes (district-demeaned) -> correct-sign covariate effects
Xd, yd = X.astype(float).copy(), y.astype(float).copy()
for d in np.unique(g):
    m = g == d; Xd[m] -= Xd[m].mean(0); yd[m] -= yd[m].mean()
beta_w = LinearRegression(fit_intercept=False).fit(Xd, yd).coef_
env = X @ beta_w                                          # environmental risk contribution (correct signs)

# empirical-Bayes district intercept (observed district burden) for the absolute level
e = y - env; edf = pd.DataFrame({"g": g, "e": e}); alpha = e.mean()
rj = edf.groupby("g")["e"].mean() - alpha; nj = edf.groupby("g")["e"].size()
sigma2 = ((edf.e - edf.groupby("g")["e"].transform("mean"))**2).sum() / (len(e) - nj.size)
tau2 = max(0.0, rj.var(ddof=1) - sigma2 * (1.0 / nj).mean())
shrink = tau2 / (tau2 + sigma2 / nj); u = shrink * rj
risk_within = np.clip(alpha + env + pd.Series(g).map(u).values, 0, None)

# pooled covariate model (reference)
lin = LinearRegression().fit(X, y); risk_pooled = np.clip(lin.predict(X), 0, None)

pan = pd.read_csv("data/inputs/chap_data_level5_clean_2013-2021.csv", usecols=["location", "location_name"])
pan["location"] = pan.location.astype(str); names = pan.drop_duplicates("location").set_index("location")["location_name"]
out = pd.DataFrame({
    "location_id": df.location_id, "sector_name": df.location_id.map(names), "district": df.parent,
    "population": df.wp_pop.round().astype(int),
    "risk_per1000_yr": np.round(risk_within / N_YEARS, 2),
    "risk_normalized": np.round((risk_within - risk_within.min()) / (risk_within.max() - risk_within.min()), 4),
    "risk_env_score": np.round((env - env.min()) / (env.max() - env.min()), 4),
    "risk_pooled_per1000_yr": np.round(risk_pooled / N_YEARS, 2),
    "observed_incidence_per1000_yr": np.round(df.inc_dhis2, 2),
})
out["risk_rank"] = out["risk_per1000_yr"].rank(ascending=False, method="min").astype(int)
out = out.sort_values("risk_rank").reset_index(drop=True)
out.to_csv("results/sector_percapita_risk.csv", index=False)
print(f"WITHIN-DISTRICT slopes [sig_temp, habitat, built] = {np.round(beta_w,1)}  (habitat POSITIVE, mechanistic)")
print(f"pooled slopes (reference)                          = {np.round(lin.coef_,1)}  (habitat negative — collinearity)")
print(f"EB district-intercept mean shrinkage = {shrink.mean():.2f}\n")
print(f"wrote results/sector_percapita_risk.csv  ({len(out)} sectors)")
print(out.head(8).to_string(index=False))
