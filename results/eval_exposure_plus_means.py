"""Linear model combining the optimal exposure index (t_opt=29) with the
pop-weighted temp / NDVI / amp means. Does the breeding-site exposure index carry
anything the simple means don't, for within-district ranking (and pooled)?

All features per-sector; LODO OOF; scored with within-district concordance
(headline), within-district mean Spearman, and pooled OOF Spearman.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr

mo = pd.read_csv("results/rwanda_sector_moisture.csv")
mo["location_id"] = mo["location_id"].astype(str)
mo = mo[mo["annual_incidence_per1000"].notna() & mo["parent"].notna()].copy()
sw = pd.read_csv("results/rwanda_sweep_temp.csv"); sw["location_id"] = sw["location_id"].astype(str)
expo = sw.groupby("location_id")["mean_exposure_per_person__expo_000"].mean().rename("exposure")  # lam1500 g100 t29
df = mo.merge(expo, on="location_id", how="left")
y = df["annual_incidence_per1000"].values
groups = df["parent"].astype(str).values
def feats(cols):
    X = df[cols].values.astype(float)
    return np.where(np.isfinite(X), X, np.nanmean(np.where(np.isfinite(X), X, np.nan), 0))

logo = LeaveOneGroupOut()
def oof(model, X):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return pred
def concordance(p):
    C = Dd = 0.0
    for d in np.unique(groups):
        i = np.where(groups == d)[0]
        for a in range(len(i)):
            for b in range(a + 1, len(i)):
                dy = y[i[a]] - y[i[b]]
                if dy == 0: continue
                dp = p[i[a]] - p[i[b]]
                if dp == 0: C += .5; Dd += .5
                elif np.sign(dp) == np.sign(dy): C += 1
                else: Dd += 1
    return C / (C + Dd)
def wmean(p, min_n=4):
    rs = [spearmanr(p[groups == d], y[groups == d]).statistic for d in np.unique(groups)
          if (groups == d).sum() >= min_n and np.ptp(p[groups == d]) > 0]
    return np.nanmean([r for r in rs if np.isfinite(r)])

MEANS = ["temp_pop", "ndvi_ann", "ndvi_amp"]
SETS = {
    "exposure only":                 ["exposure"],
    "temp only":                     ["temp_pop"],
    "means(temp,ndvi,amp)":          MEANS,
    "exposure + temp":               ["exposure", "temp_pop"],
    "exposure + means(temp,ndvi,amp)": ["exposure"] + MEANS,
}
print(f"sectors={len(y)}  districts={pd.Series(groups).nunique()}\n")
print(f"{'linear model on...':36}{'concordance':>12}{'within-mean':>12}{'pooled':>9}")
for name, cols in SETS.items():
    p = oof(LinearRegression(), feats(cols))
    print(f"{name:36}{concordance(p):>12.3f}{wmean(p):>12.3f}{spearmanr(p, y).statistic:>9.3f}")
# bonus: RF on the full combo
p = oof(RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=0, n_jobs=-1),
        feats(["exposure"] + MEANS))
print(f"{'[RF] exposure + means':36}{concordance(p):>12.3f}{wmean(p):>12.3f}{spearmanr(p, y).statistic:>9.3f}")
print("\nrefs (concordance): mean-temp 0.688 ; linear-means 0.690 ; old exposure 0.671 ; random 0.503")
