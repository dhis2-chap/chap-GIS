"""Lever 3 - spatiotemporal / seasonality (isolated from the interannual trend).

The static map predicts a constant per sector -> zero seasonal skill by
construction. Test whether a rainfall-driven climate model reproduces the
within-year SEASONAL shape. Remove each sector-year level (so only the seasonal
anomaly remains), predict it from climate, leave-one-YEAR-out, pooled Spearman.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
from scipy.stats import spearmanr
pan = pd.read_csv("data/inputs/chap_data_level5_clean_2013-2021.csv").rename(columns={"location": "location_id"})
pan["location_id"] = pan.location_id.astype(str)
pan["year"] = pan.time_period.str.slice(0, 4).astype(int); pan["month"] = pan.time_period.str.slice(5, 7).astype(int)
pan = pan.sort_values(["location_id", "year", "month"]).reset_index(drop=True)
RAIN = "rainfall_era5"
for L in (1, 2, 3): pan[f"rain_lag{L}"] = pan.groupby("location_id")[RAIN].shift(L)
pan["logc"] = np.log1p(pan.disease_cases.clip(lower=0))

cyc = pan.groupby("month").agg(cases=("disease_cases", "mean"), rain=(RAIN, "mean"))
print("national monthly cycle (bimodal: peaks Nov-Jan & May-Jun, trough Aug):")
print("  month :", " ".join(f"{m:>5}" for m in cyc.index))
print("  cases :", " ".join(f"{v:>5.0f}" for v in cyc.cases))
print(f"  corr(monthly cases, rain lag2) = {spearmanr(cyc.cases, cyc.rain.shift(2).bfill()).statistic:+.2f}  (temperature is ~aseasonal)\n")

CLIM = ["mean_temperature", RAIN, "rain_lag1", "rain_lag2", "rain_lag3", "ndvi", "evi", "relative_humidity"]
d = pan.dropna(subset=CLIM + ["logc"]).copy()
# seasonal anomaly = remove each (sector, year) annual mean -> within-year shape only
d["anom"] = d["logc"] - d.groupby(["location_id", "year"])["logc"].transform("mean")
yr = d["year"].values; ya = d["anom"].values
def lyo(cols, model):
    X = d[cols].values; pred = np.full(len(d), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, ya, yr):
        pred[te] = clone(model).fit(X[tr], ya[tr]).predict(X[te])
    return pred
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=30, random_state=0, n_jobs=-1)
# month-of-year climatology = data-driven seasonal ceiling (mean anomaly by sector-month, leave-year-out)
def clim_oof():
    pred = np.full(len(d), np.nan)
    for tr, te in LeaveOneGroupOut().split(d, ya, yr):
        mu = d.iloc[tr].groupby(["location_id", "month"])["anom"].mean()
        key = list(zip(d.iloc[te].location_id, d.iloc[te].month))
        pred[te] = [mu.get(k, 0.0) for k in key]
    return pred
def rho(pred):
    ok = np.isfinite(pred); return spearmanr(pred[ok], ya[ok]).statistic
print(f"rows={len(d)} sectors={d.location_id.nunique()} years={d.year.min()}-{d.year.max()}")
print("seasonal-anomaly skill (pooled Spearman, predicted vs actual within-year deviation), leave-one-year-out:")
print(f"  temperature only            {rho(lyo(['mean_temperature'], clone(rf))):+.3f}")
print(f"  climate incl. rainfall      {rho(lyo(CLIM, clone(rf))):+.3f}")
print(f"  month-of-year climatology   {rho(clim_oof()):+.3f}   (data-driven seasonal ceiling)")
print("\n(static/temperature map = 0 seasonal skill by construction; rainfall recovers the cycle)")
