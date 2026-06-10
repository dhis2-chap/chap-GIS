"""Does a district random effect / within-estimator improve WITHIN-district ranking?

A district random intercept does not change within-district ranking at prediction
time on a held-out district (it is a constant offset). Its only leverage is on how
the SLOPES are learned: the within (fixed-effects) estimator removes each
district's mean before fitting, so coefficients explain within-district variation
rather than the between-district gradient.

We compare, per feature set and learner:
  POOLED : fit on raw (X,y)                          [current models]
  WITHIN : fit on district-demeaned (X,y); predict the held-out district from the
           slopes only (offset irrelevant for ranking)
Metric: within-district mean OOF Spearman (the target metric), leave-one-district-out.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr

df = pd.read_csv("results/rwanda_sector_moisture.csv")
df = df[df["annual_incidence_per1000"].notna() & df["parent"].notna()].copy()
y = df["annual_incidence_per1000"].values.astype(float)
groups = df["parent"].astype(str).values

SETS = {
    "temp only":            ["temp_pop"],
    "means(temp,ndvi,amp)": ["temp_pop", "ndvi_ann", "ndvi_amp"],
    "temp+veg+moisture":    ["temp_pop", "ndvi_ann", "evi_ann", "ndvi_dry", "ndvi_drop",
                             "ndvi_ratio", "ndvi_amp", "evi_dry", "evi_drop", "evi_ratio", "evi_amp"],
}
def feats(cols):
    X = df[cols].values.astype(float)
    return np.where(np.isfinite(X), X, np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0))

logo = LeaveOneGroupOut()
def demean(X, g):
    Xd = X.copy()
    for d in np.unique(g):
        m = g == d; Xd[m] = Xd[m] - Xd[m].mean(0)
    return Xd

def oof(model, X, within):
    pred = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        if within:
            gtr = groups[tr]
            Xtr = demean(X[tr], gtr); ytr = y[tr].copy()
            for d in np.unique(gtr):
                m = gtr == d; ytr[m] = ytr[m] - ytr[m].mean()
            mdl = clone(model).fit(Xtr, ytr)
            Xte = X[te] - X[te].mean(0)            # te is one district -> its own mean
            pred[te] = mdl.predict(Xte)
        else:
            pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
    return pred

def within_skill(pred, min_n=4):
    rs = [spearmanr(pred[groups == d], y[groups == d]).statistic
          for d in np.unique(groups) if (groups == d).sum() >= min_n
          and np.ptp(pred[groups == d]) > 0]
    rs = np.array([r for r in rs if np.isfinite(r)])
    return rs.mean(), np.tanh(np.mean(np.arctanh(np.clip(rs, -.999, .999)))), spearmanr(pred, y).statistic

RF = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=0, n_jobs=-1)
print(f"sectors={len(y)} districts={pd.Series(groups).nunique()}\n")
print(f"{'feature set':22}{'learner':9}{'fit':8}{'within-mean':>12}{'within-Fisher':>14}{'pooled':>9}")
for sname, cols in SETS.items():
    X = feats(cols)
    for lname, model in (("Linear", LinearRegression()), ("RF", clone(RF))):
        for within in (False, True):
            m, f, p = within_skill(oof(model, X, within))
            print(f"{sname:22}{lname:9}{'WITHIN' if within else 'POOLED':8}{m:>12.3f}{f:>14.3f}{p:>9.3f}")
print("\nref within-district means: mean-temp control 0.513 ; 1D surface 0.487 ; old exposure(best) ~0.49")
